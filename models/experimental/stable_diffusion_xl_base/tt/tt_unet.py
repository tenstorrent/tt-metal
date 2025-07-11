# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.tt_timesteps import TtTimesteps
from models.experimental.stable_diffusion_xl_base.tt.tt_embedding import TtTimestepEmbedding

from models.experimental.stable_diffusion_xl_base.tt.tt_downblock2d import TtDownBlock2D
from models.experimental.stable_diffusion_xl_base.tt.tt_crossattndownblock2d import TtCrossAttnDownBlock2D

from models.experimental.stable_diffusion_xl_base.tt.tt_crossattnmidblock2d import TtUNetMidBlock2DCrossAttn

from models.experimental.stable_diffusion_xl_base.tt.tt_crossattnupblock2d import TtCrossAttnUpBlock2D
from models.experimental.stable_diffusion_xl_base.tt.tt_upblock2d import TtUpBlock2D

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_conv_params,
    prepare_gn_beta_gamma,
    prepare_gn_mask,
)


class TtUNet2DConditionModel(nn.Module):
    # During testing it was observed that setting conv_weights to bfloat16 + HiFi4 leads to much better image quality.
    # Other weights seem not to have as an impact on it.
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
    ):
        super().__init__()

        self.device = device
        self.model_config = model_config

        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        self.groups = 1

        self.time_proj = TtTimesteps(device, 320, True, 0, 1)
        self.add_time_proj = TtTimesteps(device, 256, True, 0, 1)

        # Initialze embeddings with attention_weights_dtype for the time being.
        self.time_embedding = TtTimestepEmbedding(
            device, state_dict, "time_embedding", linear_weights_dtype=model_config.attention_weights_dtype
        )
        self.add_embedding = TtTimestepEmbedding(
            device, state_dict, "add_embedding", linear_weights_dtype=model_config.attention_weights_dtype
        )

        self.down_blocks = []
        self.down_blocks.append(TtDownBlock2D(device, state_dict, "down_blocks.0", model_config))
        self.down_blocks.append(
            TtCrossAttnDownBlock2D(
                device,
                state_dict,
                "down_blocks.1",
                model_config,
                640,
                10,
                640,
                True,
            )
        )
        self.down_blocks.append(
            TtCrossAttnDownBlock2D(
                device,
                state_dict,
                "down_blocks.2",
                model_config,
                1280,
                20,
                1280,
                False,
            )
        )

        self.mid_block = TtUNetMidBlock2DCrossAttn(
            device,
            state_dict,
            "mid_block",
            model_config,
            1280,
            20,
            1280,
        )

        self.up_blocks = []
        self.up_blocks.append(
            TtCrossAttnUpBlock2D(
                device,
                state_dict,
                "up_blocks.0",
                model_config,
                1280,
                20,
                1280,
                True,
            )
        )
        self.up_blocks.append(
            TtCrossAttnUpBlock2D(
                device,
                state_dict,
                "up_blocks.1",
                model_config,
                640,
                10,
                640,
                True,
            )
        )
        self.up_blocks.append(TtUpBlock2D(device, state_dict, "up_blocks.2", model_config))

        conv_weights_in = state_dict["conv_in.weight"]
        conv_bias_in = state_dict["conv_in.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        norm_weights_out = state_dict["conv_norm_out.weight"]
        norm_bias_out = state_dict["conv_norm_out.bias"]

        conv_weights_out = state_dict["conv_out.weight"]
        conv_bias_out = state_dict["conv_out.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        self.conv_output_dtype = model_config.get_conv_output_dtype()
        self.conv1_config = model_config.get_conv_config(conv_path="conv_in")
        (
            self.compute1_config,
            self.tt_conv1_weights,
            self.tt_conv1_bias,
            self.conv1_params,
        ) = prepare_conv_params(
            device,
            conv_weights_in,
            conv_bias_in,
            self.conv1_config.weights_dtype,
            fp32_dest_acc_en=(self.conv1_config.weights_dtype == ttnn.bfloat8_b)
            and (self.conv1_config.shard_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        )

        self.conv2_config = model_config.get_conv_config(conv_path="conv_out")
        (
            self.compute2_config,
            self.tt_conv2_weights,
            self.tt_conv2_bias,
            self.conv2_params,
        ) = prepare_conv_params(
            device,
            conv_weights_out,
            conv_bias_out,
            self.conv2_config.weights_dtype,
            fp32_dest_acc_en=(self.conv2_config.weights_dtype == ttnn.bfloat8_b)
            and (self.conv2_config.shard_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        )

        self.norm_core_grid = ttnn.CoreGrid(y=8, x=8)
        self.norm_groups = 32
        self.norm_eps = 1e-5

        self.gamma_t, self.beta_t = prepare_gn_beta_gamma(
            device, norm_weights_out, norm_bias_out, self.norm_core_grid.y
        )
        self.input_mask = prepare_gn_mask(
            self.device, norm_weights_out.shape[0], self.norm_groups, self.norm_core_grid.y
        )

    def forward(self, sample, input_shape, timestep, encoder_hidden_states, time_ids, text_embeds):
        B, C, H, W = input_shape

        temb = self.time_proj.forward(timestep)
        temb = self.time_embedding.forward(temb)

        temb_add = self.add_time_proj.forward(time_ids)
        temb_add = ttnn.to_layout(temb_add, ttnn.ROW_MAJOR_LAYOUT)
        temb_add = ttnn.reshape(temb_add, (text_embeds.shape[0], -1))
        temb_add = ttnn.concat([text_embeds, temb_add], -1)
        temb_add = ttnn.to_layout(temb_add, ttnn.TILE_LAYOUT)
        temb_add = self.add_embedding.forward(temb_add)

        temb = ttnn.add(temb, temb_add, use_legacy=False)
        ttnn.deallocate(temb_add)

        [sample, [H, W], [self.tt_conv1_weights, self.tt_conv1_bias]] = ttnn.conv2d(
            input_tensor=sample,
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

        sample = ttnn.to_memory_config(sample, ttnn.DRAM_MEMORY_CONFIG)
        residuals = (sample,)

        temb = ttnn.typecast(temb, dtype=ttnn.bfloat16)

        ttnn.DumpDeviceProfiler(self.device)
        for i, down_block in enumerate(self.down_blocks):
            if i == 0:
                sample, [C, H, W], block_residuals = down_block.forward(sample, [B, C, H, W], temb=temb)
            else:
                sample, [C, H, W], block_residuals = down_block.forward(
                    sample, [B, C, H, W], temb=temb, encoder_hidden_states=encoder_hidden_states
                )

            residuals += block_residuals
        ttnn.DumpDeviceProfiler(self.device)

        sample, [C, H, W] = self.mid_block.forward(
            sample, [B, C, H, W], temb=temb, encoder_hidden_states=encoder_hidden_states
        )
        ttnn.DumpDeviceProfiler(self.device)

        encoder_hidden_states = ttnn.to_memory_config(encoder_hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        for i, up_block in enumerate(self.up_blocks):
            block_residuals = residuals[-len(up_block.resnets) :]
            residuals = residuals[: -len(up_block.resnets)]

            if i == 2:
                sample, [C, H, W] = up_block.forward(
                    sample,
                    block_residuals,
                    [B, C, H, W],
                    temb=temb,
                )
            else:
                sample, [C, H, W] = up_block.forward(
                    sample,
                    block_residuals,
                    [B, C, H, W],
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                )

        ttnn.DumpDeviceProfiler(self.device)

        sample = ttnn.to_layout(sample, ttnn.ROW_MAJOR_LAYOUT)

        grid_coord = ttnn.CoreCoord(self.norm_core_grid.x - 1, self.norm_core_grid.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_shape = B * H * W // self.norm_core_grid.x, C // self.norm_core_grid.y
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        sample = ttnn.to_memory_config(sample, sharded_mem_config)

        sample = ttnn.group_norm(
            sample,
            num_groups=self.norm_groups,
            input_mask=self.input_mask,
            weight=self.gamma_t,
            bias=self.beta_t,
            memory_config=sharded_mem_config,
            core_grid=self.norm_core_grid,
            epsilon=self.norm_eps,
        )

        sample = ttnn.silu(sample)

        sample = ttnn.sharded_to_interleaved(sample, ttnn.L1_MEMORY_CONFIG)

        [sample, [H, W], [self.tt_conv2_weights, self.tt_conv2_bias]] = ttnn.conv2d(
            input_tensor=sample,
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

        return sample, [C, H, W]
