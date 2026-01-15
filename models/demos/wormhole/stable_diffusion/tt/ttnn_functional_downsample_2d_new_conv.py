# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import torch

import ttnn
from models.demos.wormhole.stable_diffusion.sd_helper_funcs import reshard_for_output_channels_divisibility
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import get_default_compute_config


def permute_conv_parameters(weight, bias):
    weight = ttnn.to_torch(weight)
    weight = torch.permute(weight, (2, 3, 0, 1))
    bias = ttnn.to_torch(bias)
    return weight, bias


class downsample_2d:
    def __init__(
        self,
        device,
        parameters,
        batch_size,
        input_height,
        input_width,
        compute_kernel_config,
    ):
        self.device = device
        self.parameters = parameters
        parameters.conv.weight, parameters.conv.bias = permute_conv_parameters(
            parameters.conv.weight, parameters.conv.bias
        )
        parameters.conv.bias = torch.reshape(parameters.conv.bias, (1, 1, 1, parameters.conv.bias.shape[-1]))

        self.conv_weights = ttnn.from_torch(parameters.conv.weight, ttnn.float32)
        self.conv_bias = ttnn.from_torch(parameters.conv.bias, ttnn.float32)
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size
        self.out_channels = parameters.conv.weight.shape[0]
        self.in_channels = parameters.conv.weight.shape[1]

        self.stride = 2

        self.output_height = ttnn.get_conv_output_dim(input_height, 3, self.stride, 1)
        self.output_width = ttnn.get_conv_output_dim(input_width, 3, self.stride, 1)
        self.shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if self.in_channels < 320 else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )

    def __call__(
        self,
        in_channels,
        hidden_states,
        use_conv=False,
        out_channels=None,
        padding=1,
        dtype: Optional[ttnn.DataType] = None,
    ):
        assert padding == 1

        assert hidden_states.shape[-1] == in_channels

        if use_conv and padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = ttnn.pad(hidden_states, pad, value=0)

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=self.shard_layout,
            reshard_if_not_optimal=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )

        compute_config = get_default_compute_config(self.device)

        conv_kwargs = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "batch_size": self.batch_size,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "kernel_size": (3, 3),
            "stride": (self.stride, self.stride),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": self.device,
            "conv_config": conv_config,
            "slice_config": ttnn.Conv2dL1FullSliceConfig,
        }

        hidden_states, [self.conv_weights, self.conv_bias] = ttnn.conv2d(
            input_tensor=hidden_states,
            **conv_kwargs,
            weight_tensor=self.conv_weights,
            bias_tensor=self.conv_bias,
            compute_config=compute_config,
            dtype=ttnn.bfloat8_b,
            return_weights_and_bias=True,
        )
        hidden_states = reshard_for_output_channels_divisibility(hidden_states, self.out_channels)

        return hidden_states
