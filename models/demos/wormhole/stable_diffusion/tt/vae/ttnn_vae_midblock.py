# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_attention import Attention
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_resnet import ResnetBlock


class MidBlock:
    def __init__(
        self,
        torch_midblock,
        device,
        in_channels,
        input_height,
        input_width,
        resnet_norm_num_blocks=[(1, 1), (1, 1)],
        resnet_conv_channel_split_factors=[((1, 1), (1, 1)), ((1, 1), (1, 1))],
    ):
        self.resnets = []
        self.resnets.append(
            ResnetBlock(
                torch_midblock.resnets[0],
                device,
                in_channels,
                input_height,
                input_width,
                in_channels,
                resnet_norm_num_blocks[0][0],
                resnet_norm_num_blocks[0][1],
                resnet_conv_channel_split_factors[0][0],
                resnet_conv_channel_split_factors[0][1],
            )
        )
        self.attention = Attention(
            torch_midblock.attentions[0],
            device,
            in_channels,
        )

        self.resnets.append(
            ResnetBlock(
                torch_midblock.resnets[1],
                device,
                in_channels,
                input_height,
                input_width,
                in_channels,
                resnet_norm_num_blocks[1][0],
                resnet_norm_num_blocks[1][1],
                resnet_conv_channel_split_factors[1][0],
                resnet_conv_channel_split_factors[1][1],
            )
        )

    def __call__(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states
