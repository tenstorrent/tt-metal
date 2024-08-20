# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
import torch

import ttnn
from tt_lib.fallback_ops import fallback_ops

from models.experimental.stable_diffusion.tt.residual_block import TtResnetBlock2D
from models.experimental.stable_diffusion.tt.downsample_2d import TtDownsample2D


class TtDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        state_dict=None,
        base_address=None,
    ):
        super().__init__()
        self.state_dict = state_dict
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                TtResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    state_dict=self.state_dict,
                    base_address=f"{base_address}.resnets.{i}",
                )
            )

        self.resnets = resnets

        if add_downsample:
            self.downsamplers = [
                TtDownsample2D(
                    out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                    name="op",
                    state_dict=self.state_dict,
                    base_address=f"{base_address}.downsamplers.0",
                )
            ]
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states: ttnn.Tensor, temb: Optional[ttnn.Tensor]) -> ttnn.Tensor:
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states
