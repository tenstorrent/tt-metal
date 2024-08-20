# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
from models.experimental.ssd.tt.ssd_mobilenetv3_convlayer import (
    TtMobileNetV3ConvLayer,
)


class TtMobileNetV3Stem(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        expanded_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        stride: int = 1,
        state_dict=None,
        base_address="",
        device=None,
    ) -> None:
        super().__init__()

        self.conv_3x3 = TtMobileNetV3ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            use_activation=True,
            groups=expanded_channels,
            state_dict=state_dict,
            base_address=f"{base_address}.block.0",
            device=device,
        )

        self.reduce_1x1 = TtMobileNetV3ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            state_dict=state_dict,
            base_address=f"{base_address}.block.1",
            device=device,
        )

    def forward(self, features: ttnn.Tensor) -> ttnn.Tensor:
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)
        return features
