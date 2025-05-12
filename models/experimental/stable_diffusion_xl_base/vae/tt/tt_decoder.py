# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_midblock2d import TtUNetMidBlock2D
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_upblock2d import TtUpDecoderBlock2D
from models.experimental.stable_diffusion_xl_base.vae.tt.vae_utility import get_DRAM_conv_config, get_DRAM_GN_config
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_conv_params,
    prepare_gn_beta_gamma,
    prepare_gn_mask,
)
from loguru import logger


class TtDecoder(nn.Module):
    def __init__(self, device, state_dict):
        super().__init__()

        self.device = device

        self.norm_groups = 32
        self.norm_eps = 1e-5

        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        self.groups = 1

        num_up_blocks = 4

        self.mid_block = TtUNetMidBlock2D(device, state_dict, "decoder.mid_block")
        self.up_blocks = []
        for block_id in range(num_up_blocks):
            self.up_blocks.append(
                TtUpDecoderBlock2D(
                    device,
                    state_dict,
                    f"decoder.up_blocks.{block_id}",
                    has_upsample=block_id < 3,
                    conv_shortcut=block_id > 1,
                )
            )

        norm_out_weights = state_dict[f"decoder.conv_norm_out.weight"]
        norm_out_bias = state_dict[f"decoder.conv_norm_out.bias"]

        conv_in_weights = state_dict[f"decoder.conv_in.weight"]
        conv_in_bias = state_dict[f"decoder.conv_in.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        self.conv_out_weights = state_dict[f"decoder.conv_out.weight"]
        self.conv_out_bias = state_dict[f"decoder.conv_out.bias"]

        core_y, self.norm_blocks = get_DRAM_GN_config(None, 1)
        self.norm_core_grid = ttnn.CoreGrid(y=core_y, x=8)

        self.gamma_t, self.beta_t = prepare_gn_beta_gamma(
            device, norm_out_weights, norm_out_bias, self.norm_core_grid.y
        )
        self.input_mask = prepare_gn_mask(
            self.device, norm_out_weights.shape[0], self.norm_groups, self.norm_core_grid.y
        )

        (
            self.compute_config,
            self.conv_config,
            self.tt_conv_in_weights,
            self.tt_conv_in_bias,
            self.conv_in_params,
        ) = prepare_conv_params(
            device,
            conv_in_weights,
            conv_in_bias,
            ttnn.bfloat16,
            act_block_h_override=32,
            fp32_dest_acc_en=True,
            math_fidelity=ttnn.MathFidelity.LoFi,
        )
        self.conv_in_slice_config = get_DRAM_conv_config(None, 1)

    def forward(self, sample, input_shape):
        B, C, H, W = input_shape
        hidden_states = sample

        if self.conv_in_slice_config is not None:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (B, H, W, C))
        [hidden_states, [H, W], [d_w, d_b]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.tt_conv_in_weights,
            in_channels=self.conv_in_params["input_channels"],
            out_channels=self.conv_in_params["output_channels"],
            device=self.device,
            bias_tensor=self.tt_conv_in_bias,
            kernel_size=self.conv_in_params["kernel_size"],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            memory_config=None,
            slice_config=self.conv_in_slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        C = self.conv_in_params["output_channels"]

        self.tt_conv_in_weights = d_w
        self.tt_conv_in_bias = d_b

        if self.conv_in_slice_config is not None:
            hidden_states = ttnn.reshape(hidden_states, (1, 1, B * H * W, C))
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        logger.info("Starting mid-block")
        hidden_states, [C, H, W] = self.mid_block.forward(hidden_states, [B, C, H, W])

        for idx, up_block in enumerate(self.up_blocks):
            logger.info(f"Starting {idx}. up-block")
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
            hidden_states, [C, H, W] = up_block.forward(hidden_states, [B, C, H, W])

        logger.info("Executing out ops")
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=self.input_mask,
            weight=self.gamma_t,
            bias=self.beta_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=self.norm_core_grid,
            epsilon=self.norm_eps,
            inplace=False,
            num_out_blocks=self.norm_blocks,
        )

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.silu(hidden_states)

        # HOST FALLBACK: Conv2d
        hidden_states = ttnn.to_torch(hidden_states).float()
        hidden_states = hidden_states.reshape(B, H, W, C)
        hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))

        hidden_states = F.conv2d(hidden_states, self.conv_out_weights, bias=self.conv_out_bias, stride=1, padding=1)
        return hidden_states
