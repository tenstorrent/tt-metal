# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_configs import (
    get_default_compute_config,
    get_default_conv_config,
    get_default_conv_output_dtype,
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
        kernel_size=3,
        padding=1,
    ):
        self.device = device
        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.return_weights_and_bias = True
        self.compute_config = get_default_compute_config(device)
        self.conv_config = get_default_conv_config()
        self.conv_config.enable_act_double_buffer = True
        self.conv_config.enable_weights_double_buffer = True
        self.conv_output_dtype = get_default_conv_output_dtype()
        self.conv_weights = ttnn.from_torch(torch_conv.weight)
        self.conv_bias = ttnn.from_torch(torch_conv.bias[None, None, None, :])

    def __call__(
        self,
        hidden_states,
    ):
        conv_kwargs = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "batch_size": 1,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "kernel_size": (self.kernel_size, self.kernel_size),
            "stride": (1, 1),
            "padding": (self.padding, self.padding),
            "dilation": (1, 1),
            "groups": 1,
            "device": self.device,
            "conv_config": self.conv_config,
        }

        hidden_states, [self.conv_weights, self.conv_bias] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.conv_weights,
            bias_tensor=self.conv_bias,
            **conv_kwargs,
            compute_config=self.compute_config,
            return_weights_and_bias=True,
            dtype=self.conv_output_dtype,
        )

        return hidden_states
