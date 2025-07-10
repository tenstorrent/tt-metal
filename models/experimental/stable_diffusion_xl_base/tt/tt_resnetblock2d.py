# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_gn_mask,
    prepare_gn_beta_gamma,
    prepare_conv_params,
    prepare_split_conv_params,
    split_conv2d,
    prepare_linear_params,
)


class TtResnetBlock2D(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
        conv_shortcut=False,
        split_in=1,
        split_out=1,
    ):
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
            conv_weights_3 = state_dict[f"{module_path}.conv_shortcut.weight"].squeeze()
            conv_bias_3 = state_dict[f"{module_path}.conv_shortcut.bias"]

        if split_in > 1:
            self.norm_1_blocks = 6 if "up_blocks.2.resnets.0" in module_path else 3
            core_x = core_y = 2 if "up_blocks.2.resnets.0" in module_path else 4
            self.norm_core_grid_1 = ttnn.CoreGrid(y=core_y, x=core_x)
            self.gamma_t_1, self.beta_t_1 = prepare_gn_beta_gamma(
                device, norm_weights_1, norm_bias_1, self.norm_core_grid_1.y
            )
            self.input_mask_1 = prepare_gn_mask(
                self.device, norm_weights_1.shape[0], self.norm_groups, self.norm_core_grid_1.y
            )
        else:
            self.norm_1_blocks = 2
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

        self.conv_output_dtype = model_config.get_conv_output_dtype()
        self.conv1_config = model_config.get_conv_config(conv_path=f"{module_path}.conv1")
        if self.split_conv:
            (
                self.compute1_config,
                self.tt_conv1_weights,
                self.tt_conv1_bias,
                self.conv1_params,
            ) = prepare_split_conv_params(
                device,
                conv_weights_1,
                conv_bias_1,
                self.conv1_config.weights_dtype,
                split_in,
                split_out,
                fp32_dest_acc_en=(self.conv1_config.weights_dtype == ttnn.bfloat8_b)
                and (self.conv1_config.shard_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            )
        else:
            (
                self.compute1_config,
                self.tt_conv1_weights,
                self.tt_conv1_bias,
                self.conv1_params,
            ) = prepare_conv_params(
                device,
                conv_weights_1,
                conv_bias_1,
                self.conv1_config.weights_dtype,
                fp32_dest_acc_en=(self.conv1_config.weights_dtype == ttnn.bfloat8_b)
                and (self.conv1_config.shard_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            )

        self.conv2_config = model_config.get_conv_config(conv_path=f"{module_path}.conv2")
        (
            self.compute2_config,
            self.tt_conv2_weights,
            self.tt_conv2_bias,
            self.conv2_params,
        ) = prepare_conv_params(
            device,
            conv_weights_2,
            conv_bias_2,
            self.conv2_config.weights_dtype,
            fp32_dest_acc_en=(self.conv2_config.weights_dtype == ttnn.bfloat8_b)
            and (self.conv2_config.shard_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        )

        if conv_shortcut:
            self.tt_conv3_weights, self.tt_conv3_bias = prepare_linear_params(
                device, conv_weights_3, conv_bias_3, model_config.conv_w_dtype
            )
            self.conv3_program_config = model_config.get_matmul_config(matmul_path=f"{module_path}.conv_shortcut")
        else:
            self.tt_conv3_weights = self.tt_conv3_bias = None

        self.tt_time_emb_weights, self.tt_time_emb_bias = prepare_linear_params(
            device, time_emb_weights, time_emb_bias, model_config.conv_w_dtype
        )

        mm_path = f"{module_path}.linear"
        self.linear_program_config = model_config.get_matmul_config(matmul_path=f"{module_path}.linear")
        assert self.linear_program_config is not None, "linear_program_config should not be None"
        self.default_compute_config = model_config.get_mm_compute_config(mm_path)

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
                num_out_blocks=self.norm_1_blocks,
            )
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        else:
            hidden_states = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
            grid_coord = ttnn.CoreCoord(self.norm_core_grid_1.x - 1, self.norm_core_grid_1.y - 1)
            shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
            shard_shape = B * H * W // self.norm_core_grid_1.x, C // self.norm_core_grid_1.y
            shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
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
        if hidden_states.memory_config().memory_layout != self.conv1_config.shard_layout:
            hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)

        if self.split_conv:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
            hidden_states, [C, H, W], [self.tt_conv1_weights, self.tt_conv1_bias] = split_conv2d(
                device=self.device,
                hidden_states=hidden_states,
                input_shape=[B, C, H, W],
                conv_weights=self.tt_conv1_weights,
                conv_bias=self.tt_conv1_bias,
                split_in=self.split_in,
                split_out=self.split_out,
                compute_config=self.compute1_config,
                conv_config=self.conv1_config,
                conv_params=self.conv1_params,
                conv_dtype=self.conv_output_dtype,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
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
                return_output_dim=True,
                return_weights_and_bias=True,
                dtype=self.conv_output_dtype,
            )
            C = self.conv1_params["output_channels"]

        temb = ttnn.silu(temb)

        temb = ttnn.linear(
            temb,
            self.tt_time_emb_weights,
            bias=self.tt_time_emb_bias,
            program_config=self.linear_program_config,
            compute_kernel_config=self.default_compute_config,
        )

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
        # Note: moving this add to NG has perf impact, to be investigated
        hidden_states = ttnn.add(hidden_states, temb, use_legacy=True)

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        grid_coord = ttnn.CoreCoord(self.norm_core_grid_2.x - 1, self.norm_core_grid_2.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_shape = B * H * W // self.norm_core_grid_2.x, C // self.norm_core_grid_2.y
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
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
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.conv_output_dtype,
        )
        C = self.conv2_params["output_channels"]

        if self.tt_conv3_weights is not None:
            input_tensor_pre_conv = input_tensor
            input_tensor = ttnn.linear(
                input_tensor,
                self.tt_conv3_weights,
                bias=self.tt_conv3_bias,
                program_config=self.conv3_program_config,
                compute_kernel_config=self.default_compute_config,
                memory_config=ttnn.L1_MEMORY_CONFIG
                if C == 320 and (input_shape[1] == 960 or input_shape[1] == 640)
                else hidden_states.memory_config(),
            )
            ttnn.deallocate(input_tensor_pre_conv)
            if input_tensor.memory_config() != hidden_states.memory_config():
                input_tensor = ttnn.to_memory_config(input_tensor, memory_config=hidden_states.memory_config())
        else:
            if input_tensor.is_sharded():
                input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)

            hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
        # Note: Moving this to NG results in error caused by shard shape, to be investigated
        ttnn.add_(hidden_states, input_tensor, use_legacy=True)
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

        return hidden_states, [C, H, W]
