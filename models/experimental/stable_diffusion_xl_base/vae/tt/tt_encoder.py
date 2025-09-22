# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_midblock2d import TtUNetMidBlock2D
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_downblock2d import TtDownEncoderBlock2D
from models.experimental.stable_diffusion_xl_base.vae.tt.vae_utility import get_DRAM_conv_config, get_DRAM_GN_config
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_conv_params,
    prepare_gn_beta_gamma,
    prepare_gn_mask,
)
from loguru import logger


class TtEncoder(LightweightModule):
    def __init__(self, device, state_dict, model_config):
        super().__init__()

        self.device = device

        self.norm_groups = 32
        self.norm_eps = 1e-6

        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        self.groups = 1

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
                )
            )

        norm_out_weights = state_dict[f"encoder.conv_norm_out.weight"]
        norm_out_bias = state_dict[f"encoder.conv_norm_out.bias"]

        conv_in_weights = state_dict[f"encoder.conv_in.weight"]
        conv_in_bias = state_dict[f"encoder.conv_in.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        conv_out_weights = state_dict[f"encoder.conv_out.weight"]
        conv_out_bias = state_dict[f"encoder.conv_out.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        core_x, core_y, self.norm_blocks = get_DRAM_GN_config("encoder", 1)
        self.norm_core_grid = ttnn.CoreGrid(y=core_y, x=core_x)
        self.is_sharded_gn = self.norm_blocks == -1

        if self.is_sharded_gn:
            self.gamma_t, self.beta_t = prepare_gn_beta_gamma(
                device, norm_out_weights, norm_out_bias, self.norm_core_grid.x
            )
            self.input_mask = prepare_gn_mask(
                self.device, norm_out_weights.shape[0], self.norm_groups, self.norm_core_grid.x
            )
        else:
            [self.gamma_t, self.beta_t], self.input_mask = ttnn.dram_group_norm_params_from_torch(
                [norm_out_weights, norm_out_bias],
                norm_out_weights.shape[0],
                self.norm_groups,
                device,
                core_grid=self.norm_core_grid,
                return_mask=True,
            )

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
        self.conv_in_slice_config = get_DRAM_conv_config("encoder", 1)

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
        self.conv_out_slice_config = get_DRAM_conv_config("encoder", 2)
        self.conv_output_dtype = model_config.get_conv_output_dtype()

    def forward(self, sample, input_shape):
        B, C, H, W = input_shape
        hidden_states = sample

        [hidden_states, [H, W], [self.tt_conv_in_weights, self.tt_conv_in_bias]] = ttnn.conv2d(
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

        for idx, down_block in enumerate(self.down_blocks):
            logger.info(f"Starting {idx}. down-block")
            hidden_states, [C, H, W] = down_block.forward(hidden_states, [B, C, H, W])

        logger.info("Starting mid-block")
        hidden_states, [C, H, W] = self.mid_block.forward(hidden_states, [B, C, H, W])

        logger.info("Executing out ops")
        if self.is_sharded_gn:
            shard_shape = B * H * W // self.norm_core_grid.x, C // self.norm_core_grid.y
            sharded_mem_config = ttnn.create_sharded_memory_config(
                shard_shape,
                core_grid=self.norm_core_grid,
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            hidden_states = ttnn.to_memory_config(hidden_states, sharded_mem_config)

            hidden_states = ttnn.group_norm(
                hidden_states,
                num_groups=self.norm_groups,
                input_mask=self.input_mask,
                weight=self.gamma_t,
                bias=self.beta_t,
                memory_config=sharded_mem_config,
                core_grid=self.norm_core_grid,
                epsilon=self.norm_eps,
                negative_mask=None,
                inplace=False,  # We are working with tiled sharded GN
            )

            if self.conv_out_slice_config is not None:
                hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        else:
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

        hidden_states = ttnn.silu(hidden_states)

        [hidden_states, [H, W], [self.tt_conv_out_weights, self.tt_conv_out_bias]] = ttnn.conv2d(
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

        return hidden_states, [C, H, W]
