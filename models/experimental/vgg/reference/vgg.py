# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import cast, Dict, List, Union

import torch
import torch.nn as nn


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5,
        state_dict=None,
        base_address="",
    ) -> None:
        super().__init__()
        self.state_dict = state_dict
        self.base_address = base_address

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 0
            nn.ReLU(True),  # 1
            nn.Dropout(p=dropout),  # 2
            nn.Linear(4096, 4096),  # 3
            nn.ReLU(True),  # 4
            nn.Dropout(p=dropout),  # 5
            nn.Linear(4096, num_classes),  # 6
        )

        self.classifier[0].weight = nn.Parameter(state_dict["classifier.0.weight"])
        self.classifier[0].bias = nn.Parameter(state_dict["classifier.0.bias"])

        self.classifier[3].weight = nn.Parameter(state_dict["classifier.3.weight"])
        self.classifier[3].bias = nn.Parameter(state_dict["classifier.3.bias"])

        self.classifier[6].weight = nn.Parameter(state_dict["classifier.6.weight"])
        self.classifier[6].bias = nn.Parameter(state_dict["classifier.6.bias"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(
    cfg: List[Union[str, int]],
    batch_norm: bool = False,
    state_dict=None,
    base_address="features",
) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                conv_ind = len(layers) - 3
                bn_ind = len(layers) - 2

                conv2d.weight = nn.Parameter(state_dict[f"{base_address}.{conv_ind}.weight"])
                conv2d.bias = nn.Parameter(state_dict[f"{base_address}.{conv_ind}.bias"])

                layers[bn_ind].weight = nn.Parameter(state_dict[f"{base_address}.{bn_ind}.weight"])
                layers[bn_ind].bias = nn.Parameter(state_dict[f"{base_address}.{bn_ind}.bias"])
                layers[bn_ind].running_mean = nn.Parameter(state_dict[f"{base_address}.{bn_ind}.running_mean"])
                layers[bn_ind].running_var = nn.Parameter(state_dict[f"{base_address}.{bn_ind}.running_var"])
                layers[bn_ind].num_batches_tracked = nn.Parameter(
                    state_dict[f"{base_address}.{bn_ind}.num_batches_tracked"],
                    requires_grad=False,
                )
                layers[bn_ind].eval()
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
                ind = len(layers) - 2
                conv2d.weight = nn.Parameter(state_dict[f"{base_address}.{ind}.weight"])
                conv2d.bias = nn.Parameter(state_dict[f"{base_address}.{ind}.bias"])
            in_channels = v

    return nn.Sequential(*layers)


def _vgg(cfg: str, batch_norm: bool, state_dict) -> VGG:
    model = VGG(
        make_layers(
            cfgs[cfg],
            batch_norm=batch_norm,
            state_dict=state_dict,
        ),
        state_dict=state_dict,
    )
    return model


def vgg16(state_dict) -> VGG:
    return _vgg(
        "D",
        False,
        state_dict,
    )


def vgg16_bn(state_dict) -> VGG:
    return _vgg(
        "D",
        True,
        state_dict,
    )
