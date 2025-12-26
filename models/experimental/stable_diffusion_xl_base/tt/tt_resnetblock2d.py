# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_conv_params,
    prepare_linear_params,
)


class TtResnetBlock2D(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
        conv_shortcut=False,
        debug_mode=False,
    ):
        super().__init__()

        self.device = device
        self.module_path = module_path
        self.debug_mode = debug_mode

        # fixed for ResnetBlock
        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        self.groups = 1

        self.norm_groups = 32
        self.norm_eps = 1e-5
        self.is_first_resnet_block = "resnets.0" in module_path and "up_blocks" not in module_path

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

        (
            self.groupnorm_config_1,
            self.groupnorm_memory_config_1,
            self.input_mask_1,
            self.input_negative_mask_1,
            self.gamma_t_1,
            self.beta_t_1,
        ) = model_config.get_groupnorm_params(
            f"{module_path}.norm1", norm_weights_1, norm_bias_1, self.norm_groups, device
        )
        assert (
            self.groupnorm_memory_config_1 == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            or self.groupnorm_memory_config_1 == ttnn.DRAM_MEMORY_CONFIG
        ), "Only L1_BLOCK_SHARDED_MEMORY_CONFIG and DRAM_MEMORY_CONFIG is supported for GN"
        (
            self.groupnorm_config_2,
            self.groupnorm_memory_config_2,
            self.input_mask_2,
            self.input_negative_mask_2,
            self.gamma_t_2,
            self.beta_t_2,
        ) = model_config.get_groupnorm_params(
            f"{module_path}.norm2", norm_weights_2, norm_bias_2, self.norm_groups, device
        )
        assert (
            self.groupnorm_memory_config_2 == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            or self.groupnorm_memory_config_2 == ttnn.DRAM_MEMORY_CONFIG
        ), "Only L1_BLOCK_SHARDED_MEMORY_CONFIG and DRAM_MEMORY_CONFIG is supported for GN"

        self.conv_output_dtype = model_config.get_conv_output_dtype()
        self.conv1_config = model_config.get_conv_config(conv_path=f"{module_path}.conv1")
        self.compute1_config = model_config.get_conv_compute_config(module_path=f"{module_path}.conv1")

        (
            self.tt_conv1_weights,
            self.tt_conv1_bias,
            self.conv1_params,
        ) = prepare_conv_params(
            conv_weights_1,
            conv_bias_1,
            self.conv1_config.weights_dtype,
        )
        self.conv2_config = model_config.get_conv_config(conv_path=f"{module_path}.conv2")
        self.compute2_config = model_config.get_conv_compute_config(module_path=f"{module_path}.conv2")

        (
            self.tt_conv2_weights,
            self.tt_conv2_bias,
            self.conv2_params,
        ) = prepare_conv_params(
            conv_weights_2,
            conv_bias_2,
            self.conv2_config.weights_dtype,
        )
        if conv_shortcut:
            self.tt_conv3_weights, self.tt_conv3_bias = prepare_linear_params(
                device, conv_weights_3, conv_bias_3, model_config.ff_weights_dtype
            )
            self.conv3_program_config = model_config.get_matmul_config(matmul_path=f"{module_path}.conv_shortcut")
        else:
            self.tt_conv3_weights = self.tt_conv3_bias = None

        self.tt_time_emb_weights, self.tt_time_emb_bias = prepare_linear_params(
            device, time_emb_weights, time_emb_bias, model_config.ff_weights_dtype
        )

        mm_path = f"{module_path}.linear"
        self.linear_program_config = model_config.get_matmul_config(matmul_path=f"{module_path}.linear")
        self.default_compute_config = model_config.get_mm_compute_config(mm_path)

    def forward(self, input_tensor, temb, input_shape):
        B, C, H, W = input_shape
        hidden_states = input_tensor

        mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        if self.groupnorm_memory_config_1 == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG:
            mem_cfg = ttnn.create_sharded_memory_config(
                shape=hidden_states.shape,
                core_grid=self.groupnorm_config_1["core_grid"],
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )

            # This is an optimization to avoid unaligned DRAM/L1 sharded transfer
            if C == 320 or C == 960:
                hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)

        hidden_states = ttnn.to_memory_config(hidden_states, mem_cfg)
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=self.input_mask_1,
            negative_mask=self.input_negative_mask_1,
            weight=self.gamma_t_1,
            bias=self.beta_t_1,
            epsilon=self.norm_eps,
            memory_config=hidden_states.memory_config(),
            **self.groupnorm_config_1,
        )

        hidden_states = ttnn.silu(hidden_states, output_tensor=hidden_states)

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        [hidden_states, [H, W], [tt_conv1_weights, tt_conv1_bias]] = ttnn.conv2d(
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
            slice_config=ttnn.Conv2dL1FullSliceConfig,
            groups=self.groups,
            memory_config=None,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.conv_output_dtype,
        )
        C = self.conv1_params["output_channels"]
        if not self.debug_mode:
            self.tt_conv1_weights = tt_conv1_weights
            self.tt_conv1_bias = tt_conv1_bias

        temb = ttnn.linear(
            temb,
            self.tt_time_emb_weights,
            bias=self.tt_time_emb_bias,
            program_config=self.linear_program_config,
            compute_kernel_config=self.default_compute_config,
        )

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
        # Note: moving this add to NG has perf impact, to be investigated
        hidden_states = ttnn.add_(hidden_states, temb, use_legacy=True)

        if self.groupnorm_memory_config_2 == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG:
            mem_cfg = ttnn.create_sharded_memory_config(
                shape=hidden_states.shape,
                core_grid=self.groupnorm_config_2["core_grid"],
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)

        hidden_states = ttnn.to_memory_config(hidden_states, mem_cfg)

        if "up_blocks.2" in self.module_path:
            hidden_states = ttnn.move(hidden_states)
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=self.input_mask_2,
            negative_mask=self.input_negative_mask_2,
            weight=self.gamma_t_2,
            bias=self.beta_t_2,
            epsilon=self.norm_eps,
            memory_config=hidden_states.memory_config(),
            **self.groupnorm_config_2,
        )

        ttnn.silu(hidden_states, output_tensor=hidden_states)

        [hidden_states, [H, W], [tt_conv2_weights, tt_conv2_bias]] = ttnn.conv2d(
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
            slice_config=ttnn.Conv2dL1FullSliceConfig,
            groups=self.groups,
            memory_config=None,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.conv_output_dtype,
        )
        C = self.conv2_params["output_channels"]
        if not self.debug_mode:
            self.tt_conv2_weights = tt_conv2_weights
            self.tt_conv2_bias = tt_conv2_bias

        if self.tt_conv3_weights is not None:
            input_tensor_pre_conv = input_tensor
            input_tensor = ttnn.linear(
                input_tensor,
                self.tt_conv3_weights,
                bias=self.tt_conv3_bias,
                program_config=self.conv3_program_config,
                compute_kernel_config=self.default_compute_config,
                memory_config=ttnn.L1_MEMORY_CONFIG
                if (C == 320 and (input_shape[1] == 960 or input_shape[1] == 640))
                or (self.conv3_program_config is None)
                else hidden_states.memory_config(),
            )
            if not self.is_first_resnet_block:
                ttnn.deallocate(input_tensor_pre_conv)
        if input_tensor.memory_config() != hidden_states.memory_config():
            input_tensor = ttnn.to_memory_config(input_tensor, memory_config=hidden_states.memory_config())

        # Note: Moving this to NG results in error caused by shard shape, to be investigated
        ttnn.add_(hidden_states, input_tensor, use_legacy=True)

        if "up_blocks.2.resnets.2" not in self.module_path:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

        return hidden_states, [C, H, W]
