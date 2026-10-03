"""Model definitions for CIFAR-10 image classification."""

from torch import nn
from torchvision.models import resnet18


class ConvBlock(nn.Module):
    """Two convolution layers followed by pooling and dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class CifarCNN(nn.Module):
    """Compact CNN designed for 32x32 CIFAR-10 images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64, 0.10),
            ConvBlock(64, 128, 0.15),
            ConvBlock(128, 256, 0.20),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """Create a supported model by name."""
    name = name.lower()

    if name == "cnn":
        return CifarCNN(num_classes=num_classes)

    if name == "resnet18":
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {name}. Choose 'cnn' or 'resnet18'.")
