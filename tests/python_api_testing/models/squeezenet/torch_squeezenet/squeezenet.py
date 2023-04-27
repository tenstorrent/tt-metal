from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int, state_dict=None, base_address="") -> None:
        super().__init__()
        self.inplanes = inplanes

        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze.weight = nn.Parameter(state_dict[f"{base_address}.squeeze.weight"])
        self.squeeze.bias = nn.Parameter(state_dict[f"{base_address}.squeeze.bias"])

        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1.weight = nn.Parameter(state_dict[f"{base_address}.expand1x1.weight"])
        self.expand1x1.bias = nn.Parameter(state_dict[f"{base_address}.expand1x1.bias"])

        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3.weight = nn.Parameter(state_dict[f"{base_address}.expand3x3.weight"])
        self.expand3x3.bias = nn.Parameter(state_dict[f"{base_address}.expand3x3.bias"])

        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class SqueezeNet(nn.Module):
    def __init__(self, version: str = "1_0", num_classes: int = 1000, dropout: float = 0.5, state_dict=None, base_address="") -> None:
        super().__init__()

        self.num_classes = num_classes
        if version == "1_0":
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64, state_dict=state_dict, base_address=f"features.3"),
                Fire(128, 16, 64, 64, state_dict=state_dict, base_address=f"features.4"),
                Fire(128, 32, 128, 128, state_dict=state_dict, base_address=f"features.5"),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128, state_dict=state_dict, base_address=f"features.7"),
                Fire(256, 48, 192, 192, state_dict=state_dict, base_address=f"features.8"),
                Fire(384, 48, 192, 192, state_dict=state_dict, base_address=f"features.9"),
                Fire(384, 64, 256, 256, state_dict=state_dict, base_address=f"features.10"),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256, state_dict=state_dict, base_address=f"features.12"),
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64, state_dict=state_dict, base_address=f"features.3"),
                Fire(128, 16, 64, 64, state_dict=state_dict, base_address=f"features.4"),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128, state_dict=state_dict, base_address=f"features.6"),
                Fire(256, 32, 128, 128, state_dict=state_dict, base_address=f"features.7"),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192, state_dict=state_dict, base_address=f"features.9"),
                Fire(384, 48, 192, 192, state_dict=state_dict, base_address=f"features.10"),
                Fire(384, 64, 256, 256, state_dict=state_dict, base_address=f"features.11"),
                Fire(512, 64, 256, 256, state_dict=state_dict, base_address=f"features.12"),
            )
        else:
            raise ValueError(f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected")

        self.features[0].weight = nn.Parameter(state_dict["features.0.weight"])
        self.features[0].bias = nn.Parameter(state_dict["features.0.bias"])

        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier[1].weight = nn.Parameter(state_dict["classifier.1.weight"])
        self.classifier[1].bias = nn.Parameter(state_dict["classifier.1.bias"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _squeezenet(version: str, state_dict, ) -> SqueezeNet:

    model = SqueezeNet(version, state_dict=state_dict)

    return model


# weights: SqueezeNet1_0_Weights.IMAGENET1K_V1
def squeezenet1_0(state_dict) -> SqueezeNet:
    return _squeezenet("1_0", state_dict=state_dict)


# weights:  SqueezeNet1_1_Weights.IMAGENET1K_V1
def squeezenet1_1(state_dict) -> SqueezeNet:
    return _squeezenet("1_1", state_dict=state_dict)
