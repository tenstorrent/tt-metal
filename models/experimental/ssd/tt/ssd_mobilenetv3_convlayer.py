# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Union
import torch.nn as nn
import ttnn

import tt_lib.fallback_ops as fallback_ops
from models.utility_functions import (
    torch_to_tt_tensor_rm,
)
from models.experimental.ssd.ssd_utils import create_batchnorm

ACT_FN_1 = ttnn.relu
ACT_FN_2 = ttnn.hardswish


class TtMobileNetV3ConvLayer(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
        use_activation: Union[bool, str] = False,
        activation="",
        state_dict=None,
        base_address="",
        device=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.activation_str = activation

        weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.0.weight"], device, put_on_device=False)
        bias = None

        self.convolution = fallback_ops.Conv2d(
            weights=weight,
            biases=bias,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            dilation=dilation,
        )

        self.normalization = create_batchnorm(out_channels, state_dict, f"{base_address}.1", device)

        if use_activation:
            if activation == "HS":
                self.activation = ACT_FN_2
            else:
                self.activation = ACT_FN_1
        else:
            self.activation = None

    def forward(self, features: ttnn.Tensor) -> ttnn.Tensor:
        features = self.convolution(features)
        features = self.normalization(features)
        if self.activation is not None:
            features = self.activation(features)

        return features
