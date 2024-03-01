# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
from typing import Optional
import torch.nn as nn
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_utility_functions import (
    run_ttnn_conv_with_pre_and_post_tensor_formatting,
)
import math


def permute_conv_parameters(weight, bias):
    weight = ttnn.to_layout(weight, layout=ttnn.ROW_MAJOR_LAYOUT)
    weight = ttnn.to_torch(weight)
    weight = torch.permute(weight, (2, 3, 0, 1))
    bias = ttnn.to_layout(bias, layout=ttnn.ROW_MAJOR_LAYOUT)
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


def downsample_2d(
    in_channels,
    hidden_states,
    device,
    parameters,
    use_conv=False,
    out_channels=None,
    padding=1,
    reader_patterns_cache: Optional[dict] = None,
    dtype: Optional[ttnn.DataType] = None,
    compute_kernel_config=None,
):
    stride = 2

    parameters.conv.weight, parameters.conv.bias = permute_conv_parameters(parameters.conv.weight, parameters.conv.bias)
    conv_on_device = reader_patterns_cache is not None
    batch_size = hidden_states.shape[0]
    input_height = hidden_states.shape[2]
    input_width = hidden_states.shape[3]

    if use_conv:
        if conv_on_device:
            parameters.conv.bias = torch.reshape(parameters.conv.bias, (1, 1, 1, parameters.conv.bias.shape[-1]))
            tt_weight_tensor = ttnn.from_torch(parameters.conv.weight, ttnn.float32)
            tt_bias_tensor = ttnn.from_torch(parameters.conv.bias, ttnn.float32)
            # breakpoint()
            out_channels = parameters.conv.weight.shape[0]
            in_channels = parameters.conv.weight.shape[1]
            conv_config_override = {}
            if (out_channels, in_channels, input_height, input_width) in config_override:
                conv_config_override = config_override[(out_channels, in_channels, input_height, input_width)]
            conv = ttnn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(stride, stride),
                padding=(padding, padding),
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
                enable_auto_formatting=True,
                compute_kernel_config=compute_kernel_config,
            )
        else:
            parameters.conv.weight = torch_to_tt_tensor_rm(parameters.conv.weight, device, put_on_device=False)
            parameters.conv.bias = torch_to_tt_tensor_rm(parameters.conv.bias, device, put_on_device=False)
            conv = fallback_ops.Conv2d(
                parameters.conv.weight,
                parameters.conv.bias,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )

    else:
        assert in_channels == out_channels
        assert False, " we don't support AvgPool2d, and we should not need it either"
        conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

    assert hidden_states.shape[1] == in_channels

    if use_conv and padding == 0:
        pad = (0, 1, 0, 1)
        hidden_states = ttnn.pad(hidden_states, pad, value=0)

    assert hidden_states.shape[1] == in_channels

    if conv_on_device:
        hidden_states = run_ttnn_conv_with_pre_and_post_tensor_formatting(
            device,
            conv,
            hidden_states,
            batch_size,
            math.ceil(input_height / 2),
            math.ceil(input_width / 2),
            out_channels,
        )
    else:
        hidden_states = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)))
        hidden_states = torch_to_tt_tensor_rm(hidden_states, device)
        hidden_states = conv(hidden_states)
        hidden_states = tt_to_torch_tensor(hidden_states)

        hidden_states = ttnn.to_device(
            ttnn.to_layout(ttnn.from_torch(hidden_states, ttnn.bfloat16), ttnn.TILE_LAYOUT),
            device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return hidden_states
