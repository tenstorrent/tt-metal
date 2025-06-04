# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_conv_block import ConvBlock


class UpsampleBlock:
    def __init__(
        self,
        torch_upsample,
        device,
        in_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        conv_in_channel_split_factor,
        conv_out_channel_split_factor,
        scale_factor=2,
    ):
        self.device = device
        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width
        self.scale_factor = scale_factor

        self.conv = ConvBlock(
            torch_upsample.conv,
            device,
            in_channels,
            output_height,
            output_width,
            out_channels,
            conv_in_channel_split_factor,
            conv_out_channel_split_factor,
        )

    def __call__(self, hidden_states):
        # Prepare upsample op
        if hidden_states.layout == ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        if hidden_states.shape[1] == 1:
            hidden_states = ttnn.reshape(hidden_states, [1, self.input_height, self.input_width, self.in_channels])

        hidden_states = ttnn.upsample(hidden_states, self.scale_factor)

        hidden_states = self.conv(hidden_states)

        return hidden_states
