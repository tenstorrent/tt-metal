# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_conv_block import ConvBlock
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_configs import GROUPNORM_EPSILON, GROUPNORM_GROUPS
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_midblock import MidBlock
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_upblock import UpDecoderBlock
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_utils import prepare_group_norm


class VaeDecoder:
    def __init__(
        self,
        torch_decoder,
        device,
        in_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        midblock_in_channels,
        midblock_norm_blocks,
        midblock_conv_channel_split_factors,
        upblock_out_channels,
        upblock_out_dimensions,
        upblock_norm_blocks,
        upblock_resnet_conv_channel_split_factors,
        upblock_upsample_conv_channel_split_factors,
        norm_num_blocks=16,
        conv_in_channel_split_factors=(1, 1),
        conv_out_channel_split_factors=(2, 1),
    ):
        self.device = device

        # conv in
        self.conv_in = ConvBlock(
            torch_decoder.conv_in,
            device,
            in_channels,
            input_height,
            input_width,
            midblock_in_channels,
            conv_in_channel_split_factors[0],
            conv_in_channel_split_factors[1],
        )

        # 1 midblock
        self.midblock = MidBlock(
            torch_decoder.mid_block,
            device,
            midblock_in_channels,
            input_height,
            input_width,
            midblock_norm_blocks,
            midblock_conv_channel_split_factors,
        )
        midblock_out_channels = midblock_in_channels

        # 4 upblocks
        upblock_out_channels.insert(0, midblock_out_channels)
        upblock_out_dimensions.insert(0, input_height)
        self.upblocks = []
        for i in range(4):
            self.upblocks.append(
                UpDecoderBlock(
                    torch_decoder.up_blocks[i],
                    device,
                    upblock_out_channels[i],
                    upblock_out_dimensions[i],
                    upblock_out_dimensions[i],
                    upblock_out_channels[i + 1],
                    upblock_out_dimensions[i + 1],
                    upblock_out_dimensions[i + 1],
                    upblock_norm_blocks[i],
                    upblock_resnet_conv_channel_split_factors[i],
                    upblock_upsample_conv_channel_split_factors[i],
                )
            )

        # groupnorm
        self.norm_num_blocks = norm_num_blocks
        self.norm_grid_core = ttnn.CoreGrid(y=4, x=8)
        (
            self.norm_input_mask,
            self.norm_weights,
            self.norm_bias,
        ) = prepare_group_norm(
            self.device,
            upblock_out_channels[-1],
            self.norm_grid_core,
            torch_decoder.conv_norm_out.weight,
            torch_decoder.conv_norm_out.bias,
        )

        # conv out
        self.conv_out = ConvBlock(
            torch_decoder.conv_out,
            device,
            upblock_out_channels[-1],
            output_height,
            output_width,
            out_channels,
            conv_out_channel_split_factors[0],
            conv_out_channel_split_factors[1],
        )

    def __call__(self, hidden_states):
        hidden_states = self.conv_in(hidden_states)

        hidden_states = self.midblock(hidden_states)

        for upblock in self.upblocks:
            hidden_states = upblock(hidden_states)

        hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)

        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=GROUPNORM_GROUPS,
            input_mask=self.norm_input_mask,
            weight=self.norm_weights,
            bias=self.norm_bias,
            core_grid=self.norm_grid_core,
            dtype=ttnn.bfloat16,
            inplace=False,
            num_out_blocks=self.norm_num_blocks,
            epsilon=GROUPNORM_EPSILON,
        )

        hidden_states = ttnn.silu(hidden_states)

        hidden_states = self.conv_out(hidden_states)

        return hidden_states
