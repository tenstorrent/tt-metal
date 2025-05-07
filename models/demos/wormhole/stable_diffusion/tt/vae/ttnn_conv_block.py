# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_configs import (
    get_default_compute_config,
    get_default_conv_config,
)
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_utils import (
    prepare_split_conv_weights_bias,
    split_conv_and_run,
)


# Conv op wrapper for weights and bias caching.
# ConvBlock passes in host weights and bias to conv2d op,
# and caches returned prepared weights and bias on device for next calls.
class ConvBlock:
    def __init__(
        self,
        torch_conv,
        device,
        in_channels,
        input_height,
        input_width,
        out_channels,
        conv_in_channel_split_factor=1,
        conv_out_channel_split_factor=1,
        kernel_size=3,
        padding=1,
    ):
        self.device = device
        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width
        self.out_channels = out_channels
        self.conv_in_channel_split_factor = conv_in_channel_split_factor
        self.conv_out_channel_split_factor = conv_out_channel_split_factor
        self.kernel_size = kernel_size
        self.padding = padding

        self.return_weights_and_bias = True
        self.compute_config = get_default_compute_config(device)
        self.conv_config = get_default_conv_config()

        self.conv_weights, self.conv_bias = prepare_split_conv_weights_bias(
            in_channels,
            out_channels,
            conv_in_channel_split_factor,
            conv_out_channel_split_factor,
            torch_conv.weight,
            torch_conv.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        )

    def __call__(
        self,
        hidden_states,
    ):
        conv_result = split_conv_and_run(
            hidden_states,
            self.conv_weights,
            self.conv_bias,
            self.device,
            self.in_channels,
            self.input_height,
            self.input_width,
            self.out_channels,
            self.conv_in_channel_split_factor,
            self.conv_out_channel_split_factor,
            self.compute_config,
            self.conv_config,
            self.kernel_size,
            self.padding,
            self.return_weights_and_bias,
        )

        if self.return_weights_and_bias:
            # In the first pass, we pass in weights on host,
            # so we want to keep the weights on device and reuse them
            hidden_states, self.conv_weights, self.conv_bias = conv_result
            self.return_weights_and_bias = False
        else:
            hidden_states = conv_result

        return hidden_states
