# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_gn_mask,
    prepare_gn_beta_gamma,
    prepare_conv_params,
    prepare_split_conv_params,
    split_conv2d,
)


class TtResnetBlock2D(nn.Module):
    def __init__(self, device, state_dict, module_path, conv_shortcut=False, split_in=1, split_out=1):
        super().__init__()

        self.device = device
        self.split_conv = split_in > 1 or split_out > 1
        self.split_in = split_in
        self.split_out = split_out

        # fixed for ResnetBlock
        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        self.groups = 1

        self.norm_core_grid_2 = ttnn.CoreGrid(y=8, x=8)
        self.norm_groups = 32
        self.norm_eps = 1e-5

        # loading weights
        norm_weights_1 = state_dict[f"{module_path}.norm1.weight"]
        norm_bias_1 = state_dict[f"{module_path}.norm1.bias"]

        conv_weights_1 = state_dict[f"{module_path}.conv1.weight"]
        conv_bias_1 = state_dict[f"{module_path}.conv1.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        time_emb_weights = state_dict[f"{module_path}.time_emb_proj.weight"]
        time_emb_bias = state_dict[f"{module_path}.time_emb_proj.bias"]

        norm_weights_2 = state_dict[f"{module_path}.norm2.weight"]
        norm_bias_2 = state_dict[f"{module_path}.norm2.bias"]

        conv_weights_2 = state_dict[f"{module_path}.conv2.weight"]
        conv_bias_2 = state_dict[f"{module_path}.conv2.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        if conv_shortcut:
            conv_weights_3 = state_dict[f"{module_path}.conv_shortcut.weight"]
            conv_bias_3 = state_dict[f"{module_path}.conv_shortcut.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        if split_in > 1:
            self.norm_core_grid_1 = ttnn.CoreGrid(y=4 if split_in == 2 else 2, x=8)
            self.gamma_t_1, self.beta_t_1 = prepare_gn_beta_gamma(
                device, norm_weights_1, norm_bias_1, self.norm_core_grid_1.y
            )
            self.input_mask_1 = prepare_gn_mask(
                self.device, norm_weights_1.shape[0], self.norm_groups, self.norm_core_grid_1.y
            )
        else:
            self.norm_core_grid_1 = ttnn.CoreGrid(y=8, x=8)
            self.gamma_t_1, self.beta_t_1 = prepare_gn_beta_gamma(
                device, norm_weights_1, norm_bias_1, self.norm_core_grid_1.y
            )
            self.input_mask_1 = prepare_gn_mask(
                self.device, norm_weights_1.shape[0], self.norm_groups, self.norm_core_grid_1.y
            )

        self.gamma_t_2, self.beta_t_2 = prepare_gn_beta_gamma(
            device, norm_weights_2, norm_bias_2, self.norm_core_grid_2.y
        )
        self.input_mask_2 = prepare_gn_mask(
            self.device, norm_weights_2.shape[0], self.norm_groups, self.norm_core_grid_2.y
        )

        if self.split_conv:
            (
                self.compute_config,
                self.conv_config,
                self.tt_conv1_weights,
                self.tt_conv1_bias,
                self.conv1_params,
            ) = prepare_split_conv_params(
                device, conv_weights_1, conv_bias_1, split_in, split_out, ttnn.bfloat8_b, act_block_h_override=32
            )
        else:
            (
                self.compute_config,
                self.conv_config,
                self.tt_conv1_weights,
                self.tt_conv1_bias,
                self.conv1_params,
            ) = prepare_conv_params(device, conv_weights_1, conv_bias_1, ttnn.bfloat8_b, act_block_h_override=32)
        _, _, self.tt_conv2_weights, self.tt_conv2_bias, self.conv2_params = prepare_conv_params(
            device, conv_weights_2, conv_bias_2, ttnn.bfloat8_b, act_block_h_override=32
        )
        if conv_shortcut:
            _, _, self.tt_conv3_weights, self.tt_conv3_bias, self.conv3_params = prepare_conv_params(
                device, conv_weights_3, conv_bias_3, ttnn.bfloat8_b, act_block_h_override=32
            )
        else:
            self.tt_conv3_weights = self.tt_conv3_bias = None

        self.tt_time_emb_weights = ttnn.from_torch(
            torch.permute(time_emb_weights, (1, 0)), ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT
        )
        self.tt_time_emb_bias = (
            ttnn.from_torch(time_emb_bias, ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)
            if time_emb_bias is not None
            else None
        )

    def forward(self, input_tensor, temb, input_shape):
        B, C, H, W = input_shape
        hidden_states = input_tensor

        if C >= 640 and H >= 128 and W >= 128:
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
                num_out_blocks=2,
            )
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        else:
            hidden_states = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
            grid_coord = ttnn.CoreCoord(self.norm_core_grid_1.x - 1, self.norm_core_grid_1.y - 1)
            shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
            shard_shape = B * H * W // self.norm_core_grid_1.x, C // self.norm_core_grid_1.y
            shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
            sharded_mem_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )
            hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)

            hidden_states = ttnn.group_norm(
                hidden_states,
                num_groups=self.norm_groups,
                input_mask=self.input_mask_1,
                weight=self.gamma_t_1,
                bias=self.beta_t_1,
                memory_config=sharded_mem_config,
                core_grid=self.norm_core_grid_1,
                epsilon=self.norm_eps,
            )

        hidden_states = ttnn.silu(hidden_states)
        # TBD: reshard
        if C >= 1920:
            hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
        self.conv_config.shard_layout = (
            hidden_states.memory_config().memory_layout if hidden_states.is_sharded() else None
        )
        self.conv_config.act_block_h_override = 32 if hidden_states.is_sharded() else 0

        if self.split_conv:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
            hidden_states, [C, H, W], [d_w, d_b] = split_conv2d(
                device=self.device,
                hidden_states=hidden_states,
                input_shape=[B, C, H, W],
                conv_weights=self.tt_conv1_weights,
                conv_bias=self.tt_conv1_bias,
                split_in=self.split_in,
                split_out=self.split_out,
                compute_config=self.compute_config,
                conv_config=self.conv_config,
                conv_params=self.conv1_params,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            [hidden_states, [H, W], [d_w, d_b]] = ttnn.conv2d(
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
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                groups=self.groups,
                memory_config=None,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            C = self.conv1_params["output_channels"]

        self.tt_conv1_weights = d_w
        self.tt_conv1_bias = d_b

        temb = ttnn.silu(temb)
        temb = ttnn.linear(
            temb,
            self.tt_time_emb_weights,
            bias=self.tt_time_emb_bias,
        )
        temb = ttnn.unsqueeze_to_4D(temb)
        temb = ttnn.repeat(temb, (1, 1, H * W, 1))

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.add(hidden_states, temb)

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        grid_coord = ttnn.CoreCoord(self.norm_core_grid_2.x - 1, self.norm_core_grid_2.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_shape = B * H * W // self.norm_core_grid_2.x, C // self.norm_core_grid_2.y
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)

        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=self.input_mask_2,
            weight=self.gamma_t_2,
            bias=self.beta_t_2,
            memory_config=sharded_mem_config,
            core_grid=self.norm_core_grid_2,
            epsilon=self.norm_eps,
        )

        hidden_states = ttnn.silu(hidden_states)

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
        self.conv_config.shard_layout = None
        [hidden_states, [H, W], [d_w, d_b]] = ttnn.conv2d(
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
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            memory_config=None,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        C = self.conv2_params["output_channels"]
        self.tt_conv2_weights = d_w
        self.tt_conv2_bias = d_b

        if self.tt_conv3_weights is not None:
            if input_tensor.shape[3] >= 1920:
                input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
                input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)
            self.conv_config.shard_layout = None
            self.conv_config.act_block_h_override = 0
            [input_tensor, [H, W], [d_w, d_b]] = ttnn.conv2d(
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
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                groups=self.groups,
                memory_config=None,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            C = self.conv3_params["output_channels"]
            self.tt_conv3_weights = d_w
            self.tt_conv3_bias = d_b
            if input_tensor.is_sharded():
                input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.add(input_tensor, hidden_states)

        self.conv_config.preprocess_weights_on_device = False
        self.conv_config.always_preprocess_weights = False
        return hidden_states, [C, H, W]
