# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch.nn as nn
import ttnn

from models.utility_functions import torch_to_tt_tensor_rm
from models.experimental.vovnet.vovnet_utils import create_batchnorm

from tt_lib.fallback_ops import fallback_ops


class TtConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        apply_act=True,
        norm_kwargs=None,
        act_kwargs=None,
        state_dict=None,
        base_address=None,
        device=None,
    ) -> None:
        super().__init__()
        self.device = device
        conv_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.conv.weight"], self.device, put_on_device=False)
        bias = None
        self.conv = fallback_ops.Conv2d(
            weights=conv_weight,
            biases=bias,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode="zeros",
        )

        self.bn = create_batchnorm(
            out_channels,
            state_dict,
            base_address=f"{base_address}.bn",
            device=self.device,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = ttnn.relu(x)
        return x
