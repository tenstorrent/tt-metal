# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
from typing import Optional
import torch.nn as nn
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    run_ttnn_conv_with_pre_and_post_tensor_formatting,
)
import math


def permute_conv_parameters(weight, bias):
    weight = ttnn.to_torch(weight)
    weight = torch.permute(weight, (2, 3, 0, 1))
    bias = ttnn.to_torch(bias)
    return weight, bias


config_override = {
    (320, 320, 64, 64): {"act_block_h": 64},
    (640, 640, 32, 32): {"act_block_h": 64},
    (640, 1920, 32, 32): {"act_block_h": 32},
    (640, 1280, 32, 32): {"act_block_h": 32},
    (1280, 1920, 16, 16): {"act_block_h": 32},
    (1280, 1280, 32, 32): {"act_block_h": 32},
    (320, 960, 64, 64): {"act_block_h": 32},
    (640, 960, 32, 32): {"act_block_h": 32},
    (320, 640, 64, 64): {"act_block_h": 32},
    (640, 640, 64, 64): {"act_block_h": 64},
    (640, 320, 64, 64): {"act_block_h": 64},
}


class downsample_2d:
    def __init__(
        self,
        device,
        parameters,
        reader_patterns_cache,
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
        tt_weight_tensor = ttnn.from_torch(parameters.conv.weight, ttnn.float32)
        tt_bias_tensor = ttnn.from_torch(parameters.conv.bias, ttnn.float32)

        out_channels = parameters.conv.weight.shape[0]
        in_channels = parameters.conv.weight.shape[1]
        conv_config_override = {}
        if (out_channels, in_channels, input_height, input_width) in config_override:
            conv_config_override = config_override[(out_channels, in_channels, input_height, input_width)]

        stride = 2
        self.conv = ttnn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(stride, stride),
            padding=(1, 1),
            dtype=ttnn.bfloat8_b,
            device=device,
            use_1d_systolic_array=True if in_channels < 320 else False,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            reader_patterns_cache=reader_patterns_cache,
            weight=tt_weight_tensor,
            bias=tt_bias_tensor,
            math_fidelity=ttnn.MathFidelity.LoFi,
            weights_dtype=ttnn.bfloat8_b,
            conv_blocking_and_parallelization_config_override=conv_config_override,
            use_shallow_conv_variant=False,
            # enable_auto_formatting=True,
            compute_kernel_config=compute_kernel_config,
            transpose_mcast=False,
        )

        self.output_height = self.conv.output_height
        self.output_width = self.conv.output_width

    def __call__(
        self,
        in_channels,
        hidden_states,
        use_conv=False,
        out_channels=None,
        padding=1,
        dtype: Optional[ttnn.DataType] = None,
    ):
        stride = 2
        assert padding == 1

        assert hidden_states.shape[-1] == in_channels

        if use_conv and padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = ttnn.pad(hidden_states, pad, value=0)

        if ttnn.get_memory_config(hidden_states) != self.conv.conv.input_sharded_memory_config:
            hidden_states = ttnn.to_memory_config(hidden_states, self.conv.conv.input_sharded_memory_config)
        hidden_states = self.conv(hidden_states)
        # hidden_states = run_ttnn_conv_with_pre_and_post_tensor_formatting(
        #     self.device,
        #     self.conv,
        #     hidden_states,
        #     self.conv.batch_size,
        #     self.conv.output_height,
        #     self.conv.output_width,
        #     self.conv.out_channels,
        # )

        return hidden_states
