# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_refiner.tt.tt_downblock2d import TtDownBlock2D
from models.experimental.stable_diffusion_xl_refiner.tt.tt_crossattndownblock2d import TtCrossAttnDownBlock2D
from models.experimental.stable_diffusion_xl_refiner.tt.tt_crossattnupblock2d import TtCrossAttnUpBlock2D
from models.experimental.stable_diffusion_xl_refiner.tt.tt_upblock2d import TtUpBlock2D
from models.experimental.stable_diffusion_xl_refiner.tt.tt_crossattnmidblock2d import TtCrossAttnMidBlock2D
from models.experimental.stable_diffusion_xl_base.tt.tt_timesteps import TtTimesteps
from models.experimental.stable_diffusion_xl_base.tt.tt_embedding import TtTimestepEmbedding
from models.experimental.stable_diffusion_xl_refiner.tt.components.weight_loader import WeightLoader

from models.experimental.stable_diffusion_xl_refiner.tt.components.convolution_layer import (
    ConvolutionLayer,
)
from models.experimental.stable_diffusion_xl_refiner.tt.components.group_normalization_layer import (
    GroupNormalizationLayer,
)


class TtUNet2DConditionModel(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
    ):
        super().__init__()

        self.device = device
        self.weight_loader = WeightLoader(self, state_dict)

        self.time_proj = TtTimesteps(device, 384, True, 0, 1)
        self.add_time_proj = TtTimesteps(device, 256, True, 0, 1)

        self.time_embedding = TtTimestepEmbedding(
            device, state_dict, "time_embedding", linear_weights_dtype=ttnn.bfloat16
        )
        self.add_embedding = TtTimestepEmbedding(
            device, state_dict, "add_embedding", linear_weights_dtype=ttnn.bfloat16
        )

        self.down_blocks = []
        self.down_blocks.append(
            TtDownBlock2D(
                device,
                state_dict,
                "down_blocks.0",
            )
        )
        self.down_blocks.append(
            TtCrossAttnDownBlock2D(
                device,
                state_dict,
                "down_blocks.1",
            )
        )
        self.down_blocks.append(
            TtCrossAttnDownBlock2D(
                device,
                state_dict,
                "down_blocks.2",
            )
        )

        self.down_blocks.append(
            TtDownBlock2D(
                device,
                state_dict,
                "down_blocks.3",
            )
        )

        self.mid_block = TtCrossAttnMidBlock2D(
            device,
            state_dict,
            "mid_block",
        )

        self.up_blocks = []
        self.up_blocks.append(
            TtUpBlock2D(
                device,
                state_dict,
                "up_blocks.0",
            )
        )
        self.up_blocks.append(
            TtCrossAttnUpBlock2D(
                device,
                state_dict,
                "up_blocks.1",
            )
        )
        self.up_blocks.append(
            TtCrossAttnUpBlock2D(
                device,
                state_dict,
                "up_blocks.2",
            )
        )
        self.up_blocks.append(
            TtUpBlock2D(
                device,
                state_dict,
                "up_blocks.3",
            )
        )

        self.conv_in = ConvolutionLayer(
            self.device,
            self.weight_loader.conv_in_weight,
            self.weight_loader.conv_in_bias,
        )

        self.norm = GroupNormalizationLayer(
            self.device,
            self.weight_loader.conv_norm_out_weight,
            self.weight_loader.conv_norm_out_bias,
        )

        self.conv_out = ConvolutionLayer(
            self.device,
            self.weight_loader.conv_out_weight,
            self.weight_loader.conv_out_bias,
        )

    def forward(self, hidden_states, input_shape, timestep, encoder_hidden_states, time_ids, text_embeds):
        B, C, H, W = input_shape

        # 1. time embedding
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

        # 2. initial conv
        hidden_states, [C, H, W] = self.conv_in.forward(hidden_states, B, C, H, W)

        # 3. down blocks
        down_block_res_samples = (hidden_states,)
        ttnn.ReadDeviceProfiler(self.device)
        for i, down_block in enumerate(self.down_blocks):
            if i == 0 or i == 3:
                hidden_states, [C, H, W], res_samples = down_block.forward(hidden_states, [B, C, H, W], temb)
            else:
                hidden_states, [C, H, W], res_samples = down_block.forward(
                    hidden_states, [B, C, H, W], temb, encoder_hidden_states
                )
            down_block_res_samples += res_samples
        ttnn.ReadDeviceProfiler(self.device)

        # 4. mid block
        hidden_states, [C, H, W] = self.mid_block.forward(hidden_states, [B, C, H, W], temb, encoder_hidden_states)
        ttnn.ReadDeviceProfiler(self.device)

        # 5. up blocks
        for i, up_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(up_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(up_block.resnets)]
            if i == 0 or i == 3:
                hidden_states, [C, H, W] = up_block.forward(hidden_states, [B, C, H, W], res_samples, temb)
            else:
                hidden_states, [C, H, W] = up_block.forward(
                    hidden_states, [B, C, H, W], res_samples, temb, encoder_hidden_states
                )
        ttnn.ReadDeviceProfiler(self.device)

        # 6. final norm and conv
        hidden_states = self.norm.forward(hidden_states, B, C, H, W)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.silu(hidden_states)
        hidden_states, [C, H, W] = self.conv_out.forward(hidden_states, B, C, H, W)

        return hidden_states, [C, H, W]
