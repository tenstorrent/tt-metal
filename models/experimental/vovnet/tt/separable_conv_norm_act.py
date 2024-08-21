# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch.nn as nn

import ttnn

from tt_lib import fallback_ops
from models.utility_functions import (
    torch_to_tt_tensor_rm,
)
from models.experimental.vovnet.vovnet_utils import create_batchnorm


class TtSeparableConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 1,
        bias: bool = False,
        channel_multiplier: float = 1.0,
        groups=None,
        state_dict=None,
        base_address=None,
        device=None,
    ):
        super(TtSeparableConvNormAct, self).__init__()
        self.device = device

        conv_dw_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.conv_dw.weight"],
            self.device,
            put_on_device=False,
        )
        bias = None
        self.conv_dw = fallback_ops.Conv2d(
            weights=conv_dw_weight,
            biases=bias,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            padding_mode="zeros",
        )

        conv_pw_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.conv_pw.weight"],
            self.device,
            put_on_device=False,
        )

        kernel_size = 1
        self.conv_pw = fallback_ops.Conv2d(
            weights=conv_pw_weight,
            biases=bias,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            padding_mode="zeros",
        )

        self.bn = create_batchnorm(out_channels, state_dict, base_address=f"{base_address}.bn", device=None)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        x = self.bn(x)
        x = ttnn.relu(x)
        return x
