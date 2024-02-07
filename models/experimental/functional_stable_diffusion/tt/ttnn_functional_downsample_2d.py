# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
import torch.nn as nn
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor


def permute_conv_parameters(weight, bias):
    weight = ttnn.to_layout(weight, layout=ttnn.ROW_MAJOR_LAYOUT)
    weight = ttnn.to_torch(weight)
    weight = torch.permute(weight, (2, 3, 0, 1))
    bias = ttnn.to_layout(bias, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias = ttnn.to_torch(bias)
    return weight, bias


def downsample_2d(
    in_channels,
    hidden_states,
    device,
    parameters,
    use_conv=False,
    out_channels=None,
    padding=1,
):
    stride = 2

    parameters.conv.weight, parameters.conv.bias = permute_conv_parameters(parameters.conv.weight, parameters.conv.bias)
    parameters.conv.weight = torch_to_tt_tensor_rm(parameters.conv.weight, device, put_on_device=False)
    parameters.conv.bias = torch_to_tt_tensor_rm(parameters.conv.bias, device, put_on_device=False)

    if use_conv:
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
