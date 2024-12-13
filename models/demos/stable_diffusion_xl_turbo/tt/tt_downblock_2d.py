# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from tt_lib.utils import (
    _nearest_y,
)
import math
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
from models.demos.stable_diffusion_xl_turbo.tt.tt_resnet_block_2d import ResnetBlock2D


def down_block_2d(device, parameters, config, input, temb, in_channels, input_height, input_width, num_layers=2):
    hidden_states = input
    for i in range(num_layers):
        hidden_states = ResnetBlock2D(
            config,
            hidden_states,
            temb,
            in_channels,
            input_height,
            input_width,
            parameters.resnets[i],
            device,
            conv_shortcut=False,
            use_torch_conv=False,
        )
    hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
    downsampler = run_conv(
        device,
        output_channels=parameters.downsamplers[0].conv.weight.shape[1],
        input_channels=hidden_states.shape[3],
        input_height=hidden_states.shape[1],
        input_width=hidden_states.shape[2],
        filter_height=3,
        stride_h=2,
        pad_h=1,
        tt_input_tensor=hidden_states,
        activations_dtype=ttnn.bfloat16,
        tt_weight_tensor=parameters.downsamplers[0].conv.weight,
        tt_bias_tensor=parameters.downsamplers[0].conv.bias,
    )

    return downsampler


def up_block_2d(
    device, parameters, config, input, temb, input_tuple, in_channels, input_height, input_width, num_layers=3
):
    hidden_states = input
    input_tuple = input_tuple[-1]
    for i in range(num_layers):
        hidden_states = ttnn.concat([hidden_states, input_tuple], dim=1)
        hidden_states = ResnetBlock2D(
            config,
            hidden_states,
            temb,
            in_channels=hidden_states.shape[1],
            input_height=hidden_states.shape[2],
            input_width=hidden_states.shape[3],
            parameters=parameters.resnets[i],
            device=device,
            use_torch_conv=False if i == 0 else True,
        )

    return hidden_states


def run_conv(
    device,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    stride_h,
    pad_h,
    tt_input_tensor=None,
    tt_weight_tensor=None,
    tt_bias_tensor=None,
    math_fidelity=ttnn.MathFidelity.LoFi,
    activations_dtype=ttnn.bfloat8_b,
    weights_dtype=ttnn.bfloat8_b,
    use_1d_systolic_array=True,
    config_override=None,
    use_shallow_conv_variant=False,
    dilation=1,
    fp32_accum=False,
    packer_l1_acc=False,
    output_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=False,
    debug=False,
    groups=1,
    shard_layout=None,
    auto_shard=False,
):
    batch_size = tt_input_tensor.shape[0]
    tt_weight_tensor = ttnn.to_torch(tt_weight_tensor)
    tt_weight_tensor = ttnn.from_torch(tt_weight_tensor, dtype=ttnn.float32)
    tt_bias_tensor = ttnn.to_torch(tt_bias_tensor)
    tt_bias_tensor = ttnn.from_torch(tt_bias_tensor, dtype=ttnn.float32)
    tt_bias_tensor = ttnn.reshape(tt_bias_tensor, (1, 1, 1, tt_bias_tensor.shape[0]))
    reader_patterns_cache = {}

    if shard_layout is None and not auto_shard:
        shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if use_1d_systolic_array else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=shard_layout,
        input_channels_alignment=32,
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=output_layout,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )
    if config_override and "act_block_h" in config_override and not auto_shard:
        conv_config.act_block_h_override = config_override["act_block_h"]

    if config_override and "act_block_w_div" in config_override and not auto_shard:
        conv_config.act_block_w_div = config_override["act_block_w_div"]

    if config_override and "num_cores_nhw" in config_override:
        if config_override["num_cores_nhw"] == 98:
            conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (11, 7)), ttnn.CoreRange((0, 8), (1, 8))})
            conv_config.override_sharding_config = True
    tt_output_tensor_on_device, [out_height, out_width] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_height),
        stride=(stride_h, stride_h),
        padding=(pad_h, pad_h),
        dilation=(dilation, dilation),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache=reader_patterns_cache,
        debug=debug,
        groups=groups,
        return_weights_and_bias=False,
        return_output_dim=True,
    )

    tt_output_tensor_on_device = ttnn.reshape(
        tt_output_tensor_on_device,
        (tt_output_tensor_on_device.shape[0], out_height, out_width, tt_output_tensor_on_device.shape[-1]),
    )
    tt_output_tensor_on_device = ttnn.to_torch(tt_output_tensor_on_device)
    tt_output_tensor_on_device = ttnn.from_torch(
        tt_output_tensor_on_device, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )

    tt_output_tensor_on_device = ttnn.permute(tt_output_tensor_on_device, (0, 3, 1, 2))
    return tt_output_tensor_on_device
