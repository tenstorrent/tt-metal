# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import ttnn

from models.experimental.stable_diffusion.tt.upsample_nearest2d import TtUpsampleNearest2d
from models.experimental.stable_diffusion.tt.experimental_ops import Conv2d


class TtUpsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        use_conv,
        use_conv_transpose=False,
        name="conv",
        state_dict=None,
        base_address="",
    ):
        super().__init__()
        assert not use_conv_transpose, "StableDiffusion's Unet does not use convTranspose, so leaving it out"
        self.in_channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.name = name

        self.conv = None
        if self.use_conv:
            self.conv_weight = state_dict[f"{base_address}.conv.weight"]
            self.conv_bias = state_dict[f"{base_address}.conv.bias"]
            self.conv = Conv2d(
                self.conv_weight,
                self.conv_bias,
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, hidden_states: ttnn.Tensor, output_size=None) -> ttnn.Tensor:
        assert hidden_states.shape.with_tile_padding()[1] == self.in_channels

        if output_size is None:
            upsampler_nearest2d = TtUpsampleNearest2d()
            hidden_states = upsampler_nearest2d(hidden_states)
        else:
            assert False, "we are not expected to support upsample 2d with output_size yet"
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        if self.use_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states
