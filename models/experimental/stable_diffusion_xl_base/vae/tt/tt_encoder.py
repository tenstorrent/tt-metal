# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_midblock2d import TtUNetMidBlock2D
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_downblock2d import TtDownEncoderBlock2D
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_conv_params,
)
from loguru import logger


class TtEncoder(LightweightModule):
    def __init__(self, device, state_dict, model_config, debug_mode=False):
        super().__init__()

        self.device = device

        self.norm_groups = 32
        self.norm_eps = 1e-6

        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        self.groups = 1
        self.debug_mode = debug_mode

        num_up_blocks = 4

        self.mid_block = TtUNetMidBlock2D(device, state_dict, "encoder.mid_block", model_config)
        self.down_blocks = []
        for block_id in range(num_up_blocks):
            self.down_blocks.append(
                TtDownEncoderBlock2D(
                    device,
                    state_dict,
                    f"encoder.down_blocks.{block_id}",
                    model_config,
                    has_downsample=block_id < 3,
                    has_shortcut=block_id > 0 and block_id < 3,
                    debug_mode=debug_mode,
                )
            )

        norm_out_weights = state_dict[f"encoder.conv_norm_out.weight"]
        norm_out_bias = state_dict[f"encoder.conv_norm_out.bias"]

        conv_in_weights = state_dict[f"encoder.conv_in.weight"]
        conv_in_bias = state_dict[f"encoder.conv_in.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        conv_out_weights = state_dict[f"encoder.conv_out.weight"]
        conv_out_bias = state_dict[f"encoder.conv_out.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        (
            self.groupnorm_config,
            self.groupnorm_memory_config,
            self.input_mask,
            self.input_negative_mask,
            self.gamma_t,
            self.beta_t,
        ) = model_config.get_groupnorm_params(f"encoder", norm_out_weights, norm_out_bias, self.norm_groups, device)
        assert (
            self.groupnorm_memory_config == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            or self.groupnorm_memory_config == ttnn.DRAM_MEMORY_CONFIG
        ), "Only L1_BLOCK_SHARDED_MEMORY_CONFIG and DRAM_MEMORY_CONFIG is supported for GN"

        self.compute_in_config = model_config.get_conv_compute_config(module_path="encoder.conv_in")
        self.conv_in_config = model_config.get_conv_config(conv_path="encoder.conv_in")
        (
            self.tt_conv_in_weights,
            self.tt_conv_in_bias,
            self.conv_in_params,
        ) = prepare_conv_params(
            conv_in_weights,
            conv_in_bias,
            self.conv_in_config.weights_dtype,
        )
        self.conv_in_slice_config = None

        self.compute_out_config = model_config.get_conv_compute_config(module_path="encoder.conv_out")
        self.conv_out_config = model_config.get_conv_config(conv_path="encoder.conv_out")
        (
            self.tt_conv_out_weights,
            self.tt_conv_out_bias,
            self.conv_out_params,
        ) = prepare_conv_params(
            conv_out_weights,
            conv_out_bias,
            self.conv_out_config.weights_dtype,
        )
        self.conv_out_slice_config = None
        self.conv_output_dtype = model_config.get_conv_output_dtype()

    def forward(self, sample, input_shape):
        B, C, H, W = input_shape
        hidden_states = sample

        [hidden_states, [H, W], [tt_conv_in_weights, tt_conv_in_bias]] = ttnn.conv2d(
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
            conv_config=self.conv_in_config,
            compute_config=self.compute_in_config,
            groups=self.groups,
            memory_config=None,
            slice_config=self.conv_in_slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.conv_output_dtype,
        )
        C = self.conv_in_params["output_channels"]
        if not self.debug_mode:
            self.tt_conv_in_weights = tt_conv_in_weights
            self.tt_conv_in_bias = tt_conv_in_bias

        for idx, down_block in enumerate(self.down_blocks):
            logger.info(f"Starting {idx}. down-block")
            hidden_states, [C, H, W] = down_block.forward(hidden_states, [B, C, H, W])

        logger.info("Starting mid-block")
        hidden_states, [C, H, W] = self.mid_block.forward(hidden_states, [B, C, H, W])

        logger.info("Executing out ops")
        mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        if self.groupnorm_memory_config == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG:
            mem_cfg = ttnn.create_sharded_memory_config(
                shape=hidden_states.shape,
                core_grid=self.groupnorm_config["core_grid"],
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )

        hidden_states = ttnn.to_memory_config(hidden_states, mem_cfg)
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=self.input_mask,
            negative_mask=self.input_negative_mask,
            weight=self.gamma_t,
            bias=self.beta_t,
            epsilon=self.norm_eps,
            memory_config=hidden_states.memory_config(),
            **self.groupnorm_config,
        )
        if self.conv_out_slice_config != ttnn.Conv2dL1FullSliceConfig:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

        hidden_states = ttnn.silu(hidden_states)

        [hidden_states, [H, W], [tt_conv_out_weights, tt_conv_out_bias]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.tt_conv_out_weights,
            in_channels=self.conv_out_params["input_channels"],
            out_channels=self.conv_out_params["output_channels"],
            device=self.device,
            bias_tensor=self.tt_conv_out_bias,
            kernel_size=self.conv_out_params["kernel_size"],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=self.conv_out_config,
            compute_config=self.compute_out_config,
            groups=self.groups,
            memory_config=None,
            slice_config=self.conv_out_slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.conv_output_dtype,
        )
        C = self.conv_out_params["output_channels"]
        if not self.debug_mode:
            self.tt_conv_out_weights = tt_conv_out_weights
            self.tt_conv_out_bias = tt_conv_out_bias

        return hidden_states, [C, H, W]
