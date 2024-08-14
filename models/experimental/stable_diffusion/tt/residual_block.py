# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch.nn.functional as F
import torch

import ttnn
from tt_lib.fallback_ops import fallback_ops
from models.experimental.stable_diffusion.sd_utils import make_linear
from models.experimental.stable_diffusion.tt.experimental_ops import Conv2d


class TtResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=1280,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-5,
        non_linearity="silu",
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        state_dict=None,
        base_address=None,
        host=None,
        device=None,
        use_fallback_ops=False,
    ):
        super().__init__()
        self.use_fallback_ops = use_fallback_ops
        self.pre_norm = pre_norm
        self.pre_norm = True  # this is part of the original code
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.device = device
        self.host = host
        self.out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG

        if groups_out is None:
            groups_out = groups

        norm1_weights = state_dict[f"{base_address}.norm1.weight"]
        norm1_bias = state_dict[f"{base_address}.norm1.bias"]
        self.norm1 = fallback_ops.GroupNorm(
            norm1_weights,
            norm1_bias,
            num_groups=groups,
            num_channels=self.in_channels,
            eps=eps,
            affine=True,
        )

        conv1_weights = state_dict[f"{base_address}.conv1.weight"]
        conv1_bias = state_dict[f"{base_address}.conv1.bias"]
        self.conv1 = fallback_ops.Conv2d(
            conv1_weights,
            conv1_bias,
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            time_emb_proj_weights = state_dict[f"{base_address}.time_emb_proj.weight"]
            time_emb_proj_bias = state_dict[f"{base_address}.time_emb_proj.bias"]
            self.time_emb_proj = make_linear(
                in_features=temb_channels,
                out_features=time_emb_proj_out_channels,
                weights=time_emb_proj_weights,
                bias=time_emb_proj_bias,
                device=self.device,
            )

        else:
            self.time_emb_proj = None

        norm2_weights = state_dict[f"{base_address}.norm2.weight"]
        norm2_bias = state_dict[f"{base_address}.norm2.bias"]

        self.norm2 = fallback_ops.GroupNorm(
            norm2_weights,
            norm2_bias,
            num_groups=groups,
            num_channels=self.out_channels,
            eps=eps,
            affine=True,
        )

        conv2_weights = state_dict[f"{base_address}.conv2.weight"]
        conv2_bias = state_dict[f"{base_address}.conv2.bias"]

        self.conv2 = Conv2d(
            conv2_weights,
            conv2_bias,
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if non_linearity == "swish":
            if self.use_fallback_ops:
                self.nonlinearity = fallback_ops.silu
            else:
                self.nonlinearity = ttnn.silu
        elif non_linearity == "mish":
            assert False, "Mish is not implemented!"
        elif non_linearity == "silu":
            if self.use_fallback_ops:
                self.nonlinearity = fallback_ops.silu
            else:
                self.nonlinearity = ttnn.silu

        self.upsample = self.downsample = None
        if self.up:
            assert False, "Up block within residual block is not implemented!"
        elif self.down:
            assert False, "Down block within residual block is not implemented!"

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = None
        if self.use_in_shortcut:
            conv_shortcut_weights = state_dict[f"{base_address}.conv_shortcut.weight"]
            conv_shortcut_bias = state_dict[f"{base_address}.conv_shortcut.bias"]
            self.conv_shortcut = Conv2d(
                conv_shortcut_weights,
                conv_shortcut_bias,
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, input_tensor: ttnn.Tensor, temb: ttnn.Tensor) -> ttnn.Tensor:
        out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(
            hidden_states,
        )

        if self.upsample is not None:
            assert False, "Upsample in residual block is not implemented!"
        elif self.downsample is not None:
            assert False, "Downsample in residual block is not implemented!"

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.nonlinearity(temb)

            temb = self.time_emb_proj(temb)
            temb = fallback_ops.reshape(temb, temb.get_legacy_shape()[2], temb.get_legacy_shape()[3], 1, 1)

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = ttnn.add(hidden_states, temb)

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            assert False, "Time Embedding Norm is not implemented"

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        # create a tensor of size output_scale_factor
        output_sc_recip = 1 / self.output_scale_factor
        output_sc_recip = ttnn.full(input_tensor.get_legacy_shape(), output_sc_recip)
        output_tensor = ttnn.add(input_tensor, hidden_states)
        output_tensor = ttnn.mul(output_tensor, output_sc_recip)

        return output_tensor
