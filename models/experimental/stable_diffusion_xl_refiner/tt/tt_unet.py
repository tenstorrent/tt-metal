import ttnn
from models.experimental.stable_diffusion_xl_refiner.tt.tt_downblock2d import TtDownBlock2D
from models.experimental.stable_diffusion_xl_refiner.tt.tt_crossattndownblock2d import TtCrossAttnDownBlock2D
from models.experimental.stable_diffusion_xl_refiner.tt.tt_crossattnupblock2d import TtCrossAttnUpBlock2D
from models.experimental.stable_diffusion_xl_refiner.tt.tt_upblock2d import TtUpBlock2D
from models.experimental.stable_diffusion_xl_refiner.tt.tt_crossattnmidblock2d import TtCrossAttnMidBlock2D
from models.experimental.stable_diffusion_xl_base.tt.tt_timesteps import TtTimesteps
from models.experimental.stable_diffusion_xl_base.tt.tt_embedding import TtTimestepEmbedding

from models.experimental.stable_diffusion_xl_refiner.tt.components.convolution_layer import (
    ConvolutionLayer,
    make_conv_config,
)
from models.experimental.stable_diffusion_xl_refiner.tt.components.group_normalization_layer import (
    GroupNormalizationLayer,
    make_norm_config,
)


class TtUNet2DConditionModel:
    def __init__(
        self,
        device,
        state_dict,
    ):
        super().__init__()

        self.device = device

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

        conv_weights_in = state_dict["conv_in.weight"]
        conv_bias_in = state_dict["conv_in.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        conv_in_config = make_conv_config()

        self.conv_in = ConvolutionLayer(
            self.device,
            conv_weights_in,
            conv_bias_in,
            conv_in_config,
        )

        norm_weights_out = state_dict["conv_norm_out.weight"]
        norm_bias_out = state_dict["conv_norm_out.bias"]

        norm_config = make_norm_config()

        self.norm = GroupNormalizationLayer(
            self.device,
            norm_weights_out,
            norm_bias_out,
            norm_config,
        )

        conv_weights_out = state_dict["conv_out.weight"]
        conv_bias_out = state_dict["conv_out.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        conv_out_config = make_conv_config()

        self.conv_out = ConvolutionLayer(
            self.device,
            conv_weights_out,
            conv_bias_out,
            conv_out_config,
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
        hidden_states, [C, H, W] = self.conv_in.apply(hidden_states, B, C, H, W)

        # 3. down blocks
        down_block_res_samples = (hidden_states,)
        for i, down_block in enumerate(self.down_blocks):
            print("down_block:", i, "input shape:", hidden_states.shape, "[B, C, H, W]:", [B, C, H, W])
            if i == 0 or i == 3:
                hidden_states, [C, H, W], res_samples = down_block.forward(hidden_states, [B, C, H, W], temb)
            else:
                hidden_states, [C, H, W], res_samples = down_block.forward(
                    hidden_states, [B, C, H, W], temb, encoder_hidden_states
                )
            down_block_res_samples += res_samples
            print("res samples:", res_samples)
            print("down_block_res_samples length:", len(down_block_res_samples))

        # 4. mid block
        hidden_states, [C, H, W] = self.mid_block.forward(hidden_states, [B, C, H, W], temb, encoder_hidden_states)

        # 5. up blocks
        print("Down block res samples length before up blocks:", len(down_block_res_samples))
        print("Down_block_res_samples shape:", [x.shape for x in down_block_res_samples])
        for i, up_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(up_block.resnets) :]
            print("res_samples:", res_samples)
            print
            down_block_res_samples = down_block_res_samples[: -len(up_block.resnets)]
            print("down_block_res_samples length after trim:", len(down_block_res_samples))
            if i == 0 or i == 3:
                hidden_states, [C, H, W] = up_block.forward(hidden_states, [B, C, H, W], res_samples, temb)
            else:
                hidden_states, [C, H, W] = up_block.forward(
                    hidden_states, [B, C, H, W], res_samples, temb, encoder_hidden_states
                )

        # 6. final norm and conv
        hidden_states = self.norm.apply(hidden_states, B, C, H, W)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.silu(hidden_states)
        hidden_states, [C, H, W] = self.conv_out.apply(hidden_states, B, C, H, W)

        return hidden_states, [C, H, W]
