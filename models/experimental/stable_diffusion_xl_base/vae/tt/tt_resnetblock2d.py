# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_conv_params,
    prepare_gn_beta_gamma,
    prepare_gn_mask,
)
from models.experimental.stable_diffusion_xl_base.vae.tt.vae_utility import get_DRAM_conv_config, get_DRAM_GN_config


class TtResnetBlock2D(nn.Module):
    def __init__(self, device, state_dict, module_path, model_config, conv_shortcut=False, gn_fallback=False):
        super().__init__()

        self.device = device
        self.gn_fallback = gn_fallback

        # fixed for ResnetBlock
        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        self.groups = 1

        self.norm_groups = 32
        self.norm_eps = 1e-5

        # loading weights
        self.norm_weights_1 = state_dict[f"{module_path}.norm1.weight"]
        self.norm_bias_1 = state_dict[f"{module_path}.norm1.bias"]

        conv_weights_1 = state_dict[f"{module_path}.conv1.weight"]
        conv_bias_1 = state_dict[f"{module_path}.conv1.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        self.norm_weights_2 = state_dict[f"{module_path}.norm2.weight"]
        self.norm_bias_2 = state_dict[f"{module_path}.norm2.bias"]

        conv_weights_2 = state_dict[f"{module_path}.conv2.weight"]
        conv_bias_2 = state_dict[f"{module_path}.conv2.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        if conv_shortcut:
            conv_weights_3 = state_dict[f"{module_path}.conv_shortcut.weight"]
            conv_bias_3 = state_dict[f"{module_path}.conv_shortcut.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # DEVICE CODE: GroupNorm preparation
        if not self.gn_fallback:
            core_x, core_y, self.norm_blocks_1 = get_DRAM_GN_config(module_path, 1)
            self.norm_core_grid_1 = ttnn.CoreGrid(y=core_y, x=core_x)

            self.gamma_t_1, self.beta_t_1 = prepare_gn_beta_gamma(
                device, self.norm_weights_1, self.norm_bias_1, self.norm_core_grid_1.y
            )
            self.input_mask_1 = prepare_gn_mask(
                self.device, self.norm_weights_1.shape[0], self.norm_groups, self.norm_core_grid_1.y
            )

            core_x, core_y, self.norm_blocks_2 = get_DRAM_GN_config(module_path, 2)
            self.norm_core_grid_2 = ttnn.CoreGrid(y=core_y, x=core_x)

            self.gamma_t_2, self.beta_t_2 = prepare_gn_beta_gamma(
                device, self.norm_weights_2, self.norm_bias_2, self.norm_core_grid_2.y
            )
            self.input_mask_2 = prepare_gn_mask(
                self.device, self.norm_weights_2.shape[0], self.norm_groups, self.norm_core_grid_2.y
            )

        (
            self.compute1_config,
            self.tt_conv1_weights,
            self.tt_conv1_bias,
            self.conv1_params,
        ) = prepare_conv_params(
            device,
            conv_weights_1,
            conv_bias_1,
            model_config.conv_w_dtype,
            fp32_dest_acc_en=True,
            math_fidelity=ttnn.MathFidelity.LoFi,
        )
        self.conv1_slice_config = get_DRAM_conv_config(module_path, 1)
        self.conv1_config = model_config.get_conv_config(conv_path=f"{module_path}.conv1")

        (
            self.compute2_config,
            self.tt_conv2_weights,
            self.tt_conv2_bias,
            self.conv2_params,
        ) = prepare_conv_params(
            device,
            conv_weights_2,
            conv_bias_2,
            model_config.conv_w_dtype,
            fp32_dest_acc_en=True,
            math_fidelity=ttnn.MathFidelity.LoFi,
        )
        self.conv2_slice_config = get_DRAM_conv_config(module_path, 2)
        self.conv2_config = model_config.get_conv_config(conv_path=f"{module_path}.conv2")

        if conv_shortcut:
            (
                self.compute_config_conv_linear,
                self.tt_conv3_weights,
                self.tt_conv3_bias,
                self.conv3_params,
            ) = prepare_conv_params(
                device,
                conv_weights_3,
                conv_bias_3,
                model_config.conv_w_dtype,
                fp32_dest_acc_en=False,
                math_fidelity=ttnn.MathFidelity.HiFi2,
            )
            self.conv3_config = model_config.get_conv_config(conv_path=f"{module_path}.conv_shortcut")
        else:
            self.tt_conv3_weights = self.tt_conv3_bias = None

    def forward(self, input_tensor, input_shape):
        B, C, H, W = input_shape
        # input_tensor = ttnn.reshape(input_tensor, (B, H, W, C))

        # HOST FALLBACK: GroupNorm
        if self.gn_fallback:
            hidden_states = ttnn.to_torch(input_tensor)
            hidden_states = hidden_states.reshape(B, H, W, C)
            hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
            hidden_states = F.group_norm(
                hidden_states,
                num_groups=self.norm_groups,
                weight=self.norm_weights_1,
                bias=self.norm_bias_1,
                eps=self.norm_eps,
            )
            hidden_states = torch.permute(hidden_states, (0, 2, 3, 1))
            hidden_states = hidden_states.reshape(1, 1, B * H * W, C)
            hidden_states = ttnn.from_torch(
                hidden_states,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # DEVICE CODE: GroupNorm
        else:
            input_tensor = ttnn.reshape(input_tensor, (B, 1, H * W, C))
            hidden_states = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
            hidden_states = ttnn.group_norm(
                hidden_states,
                num_groups=self.norm_groups,
                input_mask=self.input_mask_1,
                weight=self.gamma_t_1,
                bias=self.beta_t_1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                core_grid=self.norm_core_grid_1,
                epsilon=self.norm_eps,
                inplace=False,
                num_out_blocks=self.norm_blocks_1,
            )

        hidden_states = ttnn.silu(hidden_states)

        if self.conv1_slice_config is not None:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (B, H, W, C))
        [hidden_states, [H, W], [self.tt_conv1_weights, self.tt_conv1_bias]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.tt_conv1_weights,
            in_channels=self.conv1_params["input_channels"],
            out_channels=self.conv1_params["output_channels"],
            device=self.device,
            bias_tensor=self.tt_conv1_bias,
            kernel_size=self.conv1_params["kernel_size"],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=self.conv1_config,
            compute_config=self.compute1_config,
            groups=self.groups,
            memory_config=None,
            slice_config=self.conv1_slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        C = self.conv1_params["output_channels"]

        if self.conv1_slice_config is not None:
            hidden_states = ttnn.reshape(hidden_states, (1, 1, B * H * W, C))
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        # DEVICE CODE: GroupNorm
        if not self.gn_fallback:
            hidden_states = ttnn.reshape(hidden_states, (B, 1, H * W, C))
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
            hidden_states = ttnn.group_norm(
                hidden_states,
                num_groups=self.norm_groups,
                input_mask=self.input_mask_2,
                weight=self.gamma_t_2,
                bias=self.beta_t_2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                core_grid=self.norm_core_grid_2,
                epsilon=self.norm_eps,
                inplace=False,
                num_out_blocks=self.norm_blocks_2,
            )

        # HOST FALLBACK: GroupNorm
        else:
            hidden_states = ttnn.to_torch(hidden_states)
            hidden_states = hidden_states.reshape(B, H, W, C)
            hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
            hidden_states = F.group_norm(
                hidden_states,
                num_groups=self.norm_groups,
                weight=self.norm_weights_2,
                bias=self.norm_bias_2,
                eps=self.norm_eps,
            )
            hidden_states = torch.permute(hidden_states, (0, 2, 3, 1))
            hidden_states = hidden_states.reshape(1, 1, B * H * W, C)
            hidden_states = ttnn.from_torch(
                hidden_states,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.silu(hidden_states)  # note: silu hangs if not tile

        if self.conv2_slice_config is not None:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (B, H, W, C))
        [hidden_states, [H, W], [self.tt_conv2_weights, self.tt_conv2_bias]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.tt_conv2_weights,
            in_channels=self.conv2_params["input_channels"],
            out_channels=self.conv2_params["output_channels"],
            device=self.device,
            bias_tensor=self.tt_conv2_bias,
            kernel_size=self.conv2_params["kernel_size"],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=self.conv2_config,
            compute_config=self.compute2_config,
            groups=self.groups,
            memory_config=None,
            slice_config=self.conv2_slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        C = self.conv2_params["output_channels"]

        if self.conv2_slice_config is not None:
            hidden_states = ttnn.reshape(hidden_states, (1, 1, B * H * W, C))

        if self.tt_conv3_weights is not None:
            if input_tensor.shape[3] >= 1920:
                input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
                input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)
            [input_tensor, [H, W], [self.tt_conv3_weights, self.tt_conv3_bias]] = ttnn.conv2d(
                input_tensor=input_tensor,
                weight_tensor=self.tt_conv3_weights,
                in_channels=self.conv3_params["input_channels"],
                out_channels=self.conv3_params["output_channels"],
                device=self.device,
                bias_tensor=self.tt_conv3_bias,
                kernel_size=self.conv3_params["kernel_size"],
                stride=self.stride,
                padding=(0, 0),
                dilation=self.dilation,
                batch_size=input_shape[0],
                input_height=input_shape[2],
                input_width=input_shape[3],
                conv_config=self.conv3_config,
                compute_config=self.compute_config_conv_linear,
                groups=self.groups,
                memory_config=None,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            C = self.conv3_params["output_channels"]

        if input_tensor.is_sharded():
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)
        if hidden_states.is_sharded():
            hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.add(input_tensor, hidden_states)

        ttnn.deallocate(input_tensor)
        hidden_states = ttnn.move(hidden_states)

        return hidden_states, [C, H, W]
