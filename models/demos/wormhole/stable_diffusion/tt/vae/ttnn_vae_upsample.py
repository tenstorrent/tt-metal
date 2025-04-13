# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_utils import (
    get_default_compute_config,
    get_default_conv_config,
    prepare_split_conv_weights_bias,
    split_conv_and_run,
)


class UpsampleBlock:
    def __init__(
        self,
        torch_upsample,
        device,
        in_channels,
        out_channels,
        output_height,
        output_width,
        conv_in_channel_split_factor,
        conv_out_channel_split_factor,
        scale_factor=2,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_height = output_height
        self.output_width = output_width
        self.conv_in_channel_split_factor = conv_in_channel_split_factor
        self.conv_out_channel_split_factor = conv_out_channel_split_factor
        self.scale_factor = scale_factor
        self.return_weights_and_bias = True

        self.compute_config = get_default_compute_config(device)
        self.conv_config = get_default_conv_config()

        self.conv_weights, self.conv_bias = prepare_split_conv_weights_bias(
            in_channels,
            out_channels,
            conv_in_channel_split_factor,
            conv_out_channel_split_factor,
            torch_upsample.conv.weight,
            torch_upsample.conv.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        )

    def __call__(self, hidden_states):
        if hidden_states.layout == ttnn.TILE_LAYOUT:
            # Upsample op requires row-major input
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)

        hidden_states = ttnn.upsample(hidden_states, self.scale_factor)

        conv_result = split_conv_and_run(
            hidden_states,
            self.conv_weights,
            self.conv_bias,
            self.device,
            self.in_channels,
            self.output_height,
            self.output_width,
            self.out_channels,
            self.conv_in_channel_split_factor,
            self.conv_out_channel_split_factor,
            self.compute_config,
            self.conv_config,
            return_weights_and_bias=self.return_weights_and_bias,
        )

        if self.return_weights_and_bias:
            # In the first pass, we pass in weights on host,
            # so we want to keep the weights on device and reuse them
            hidden_states, self.conv_weights, self.conv_bias = conv_result
            self.return_weights_and_bias = False
        else:
            hidden_states = conv_result

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.reshape(hidden_states, [1, self.output_height, self.output_width, self.out_channels])
        return hidden_states
