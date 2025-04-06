# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
import ttnn

from models.experimental.stable_diffusion_xl_base.ttnn_impl.sdxl_utility import prepare_gn_mask, prepare_gn_beta_gamma


class TtResnetBlock2D(nn.Module):
    def __init__(self, device, state_dict, module_path, conv_shortcut=False):
        super().__init__()

        self.device = device

        # fixed for ResnetBlock
        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        self.groups = 1

        self.conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            shard_layout=None,
            input_channels_alignment=32,
            deallocate_activation=True,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=32,
            preprocess_weights_on_device=True,
            always_preprocess_weights=True,
            transpose_shards=True,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self.norm_core_grid = ttnn.CoreGrid(y=8, x=8)
        self.norm_groups = 32
        self.norm_eps = 1e-5

        # loading weights
        norm_weights_1 = state_dict[f"{module_path}.norm1.weight"]
        norm_bias_1 = state_dict[f"{module_path}.norm1.bias"]
        self.gamma_t_1, self.beta_t_1 = prepare_gn_beta_gamma(
            device, norm_weights_1, norm_bias_1, self.norm_core_grid.y
        )

        conv_weights_1 = state_dict[f"{module_path}.conv1.weight"]
        conv_bias_1 = state_dict[f"{module_path}.conv1.bias"]

        time_emb_weights = state_dict[f"{module_path}.time_emb_proj.weight"]
        time_emb_bias = state_dict[f"{module_path}.time_emb_proj.bias"]

        conv_weights_2 = state_dict[f"{module_path}.conv2.weight"]
        conv_bias_2 = state_dict[f"{module_path}.conv2.bias"]

        if conv_shortcut:
            conv_weights_3 = state_dict[f"{module_path}.conv_shortcut.weight"]
            conv_bias_3 = state_dict[f"{module_path}.conv_shortcut.bias"]

        self.tt_conv1_weights = ttnn.from_torch(conv_weights_1, ttnn.bfloat16)
        self.tt_conv1_bias = (
            ttnn.from_torch(conv_bias_1.unsqueeze(0).unsqueeze(0).unsqueeze(0), ttnn.bfloat16)
            if conv_bias_1 is not None
            else None
        )

        self.tt_time_emb_weights = ttnn.from_torch(
            torch.permute(time_emb_weights, (1, 0)), ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )
        self.tt_time_emb_bias = (
            ttnn.from_torch(time_emb_bias, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            if time_emb_bias is not None
            else None
        )

        norm_weights_2 = state_dict[f"{module_path}.norm2.weight"]
        norm_bias_2 = state_dict[f"{module_path}.norm2.bias"]
        self.gamma_t_2, self.beta_t_2 = prepare_gn_beta_gamma(
            device, norm_weights_2, norm_bias_2, self.norm_core_grid.y
        )

        self.tt_conv2_weights = ttnn.from_torch(conv_weights_2, ttnn.bfloat16)
        self.tt_conv2_bias = (
            ttnn.from_torch(conv_bias_2.unsqueeze(0).unsqueeze(0).unsqueeze(0), ttnn.bfloat16)
            if conv_bias_2 is not None
            else None
        )

        if conv_shortcut:
            self.tt_conv3_weights = ttnn.from_torch(conv_weights_3, ttnn.bfloat16)
            self.tt_conv3_bias = (
                ttnn.from_torch(conv_bias_3.unsqueeze(0).unsqueeze(0).unsqueeze(0), ttnn.bfloat16)
                if conv_bias_3 is not None
                else None
            )
        else:
            self.tt_conv3_weights = self.tt_conv3_bias = None

    def forward(self, input_tensor, temb, input_shape):
        B, C, H, W = input_shape
        hidden_states = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)

        input_mask_tensor = prepare_gn_mask(self.device, C, 32, self.norm_core_grid.y)

        grid_coord = ttnn.CoreCoord(self.norm_core_grid.x - 1, self.norm_core_grid.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_shape = B * H * W // self.norm_core_grid.x, C // self.norm_core_grid.y
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)

        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=input_mask_tensor,
            weight=self.gamma_t_1,
            bias=self.beta_t_1,
            memory_config=sharded_mem_config,
            core_grid=self.norm_core_grid,
            epsilon=self.norm_eps,
        )
        ttnn.deallocate(input_mask_tensor)

        hidden_states = ttnn.silu(hidden_states)
        # TBD: reshard
        self.conv_config.shard_layout = hidden_states.memory_config().memory_layout
        [hidden_states, [H, W], [d_w, d_b]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.tt_conv1_weights,
            in_channels=self.tt_conv1_weights.shape[1],
            out_channels=self.tt_conv1_weights.shape[0],
            device=self.device,
            bias_tensor=self.tt_conv1_bias,
            kernel_size=(self.tt_conv1_weights.shape[2], self.tt_conv1_weights.shape[3]),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            conv_op_cache={},
            debug=False,
            groups=self.groups,
            memory_config=None,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        C = self.tt_conv2_weights.shape[0]

        sig = ttnn.sigmoid(temb, memory_config=ttnn.get_memory_config(temb))
        temb = ttnn.multiply(temb, sig)
        ttnn.deallocate(sig)
        temb = ttnn.linear(
            temb,
            self.tt_time_emb_weights,
            bias=self.tt_time_emb_bias,
        )
        temb = ttnn.unsqueeze(temb, -1)
        temb = ttnn.unsqueeze(temb, -1)
        temb = ttnn.repeat(temb, (H, W))
        temb = ttnn.permute(temb, (0, 2, 3, 1))
        temb = ttnn.reshape(temb, (B, 1, H * W, C))

        hidden_states = ttnn.add(hidden_states, temb)
        ttnn.deallocate(temb)

        input_mask_tensor = prepare_gn_mask(self.device, C, 32, self.norm_core_grid.y)

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        shard_shape = B * H * W // self.norm_core_grid.x, C // self.norm_core_grid.y
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)

        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=input_mask_tensor,
            weight=self.gamma_t_2,
            bias=self.beta_t_2,
            memory_config=sharded_mem_config,
            core_grid=self.norm_core_grid,
            epsilon=self.norm_eps,
        )

        hidden_states = ttnn.silu(hidden_states)

        self.conv_config.shard_layout = hidden_states.memory_config().memory_layout
        [hidden_states, [H, W], [d_w, d_b]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.tt_conv2_weights,
            in_channels=self.tt_conv2_weights.shape[1],
            out_channels=self.tt_conv2_weights.shape[0],
            device=self.device,
            bias_tensor=self.tt_conv2_bias,
            kernel_size=(self.tt_conv2_weights.shape[2], self.tt_conv2_weights.shape[3]),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            conv_op_cache={},
            debug=False,
            groups=self.groups,
            memory_config=None,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        C = self.tt_conv2_weights.shape[0]
        if self.tt_conv3_weights is not None:
            self.conv_config.shard_layout = hidden_states.memory_config().memory_layout
            [input_tensor, [H, W], [d_w, d_b]] = ttnn.conv2d(
                input_tensor=input_tensor,
                weight_tensor=self.tt_conv3_weights,
                in_channels=self.tt_conv3_weights.shape[1],
                out_channels=self.tt_conv3_weights.shape[0],
                device=self.device,
                bias_tensor=self.tt_conv3_bias,
                kernel_size=(self.tt_conv3_weights.shape[2], self.tt_conv3_weights.shape[3]),
                stride=self.stride,
                padding=(0, 0),
                dilation=self.dilation,
                batch_size=input_shape[0],
                input_height=input_shape[2],
                input_width=input_shape[3],
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                conv_op_cache={},
                debug=False,
                groups=self.groups,
                memory_config=None,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            C = self.tt_conv3_weights.shape[0]

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.add(input_tensor, hidden_states)

        return hidden_states, [C, H, W]
