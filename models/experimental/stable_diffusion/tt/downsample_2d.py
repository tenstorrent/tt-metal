# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch.nn.functional as F
import torch

import ttnn
from tt_lib.fallback_ops import fallback_ops
from models.experimental.stable_diffusion.tt.experimental_ops import Conv2d


class TtDownsample2D(nn.Module):
    """
    A downsampling layer with an optional convolution.

    Parameters:
        channels (`int`): channels in the inputs and outputs.
        use_conv (`bool`, *optional*, defaults to False): a bool determining if a convolution is applied.
        out_channels (`int`, *optional*, defaults to channels):
        padding (`int`, *optional*, defaults to 1): padding before conv
        name (`str`, *optional*, defaults to "conv"): picks the type of conv used
        base_address (`str`, *optional*, defaults to ""): required for loading weights
        state_dict (`Dict`, *optional*, defaults to None): Dictionary of the weights
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels=None,
        padding=1,
        name="conv",
        base_address="",
        state_dict=None,
    ):
        super().__init__()
        self.base_address = base_address
        self.state_dict = state_dict
        self.in_channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv_weight = self.state_dict[f"{base_address}.conv.weight"]
            conv_bias = self.state_dict[f"{base_address}.conv.bias"]
            conv = Conv2d(
                conv_weight,
                conv_bias,
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )

        else:
            assert self.in_channels == self.out_channels
            assert False, " we don't support AvgPool2d, and we should not need it either"
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        assert hidden_states.shape.with_tile_padding()[1] == self.in_channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)

            hidden_states = fallback_ops.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape.with_tile_padding()[1] == self.in_channels
        hidden_states = self.conv(hidden_states)

        return hidden_states
