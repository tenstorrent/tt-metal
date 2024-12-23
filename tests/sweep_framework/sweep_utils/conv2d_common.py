# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import itertools
import random
import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30


def get_input_specs(
    batch_list: List[int],
    acts_list: List[int],
    kernel_list: List[int],
    stride_list: List[int],
    padding_list: List[int],
    dilation_list: List[int],
) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
    for batch_size, activation, kernel, stride, padding, dilation in itertools.product(
        batch_list, acts_list, kernel_list, stride_list, padding_list, dilation_list
    ):
        yield (batch_size, activation, activation, kernel, kernel, stride, stride, padding, padding, dilation)


def mesh_device_fixture():
    num_devices = ttnn.GetNumPCIeDevices()
    # As of now take device id as 0.
    device_id = 0
    assert device_id < num_devices, "CreateDevice not supported for non-mmio device"
    device = ttnn.CreateDevice(device_id=device_id, l1_small_size=32768)
    ttnn.SetDefaultDevice(device)

    device_name = "Unknown"
    if ttnn.device.is_grayskull(device):
        device_name = "grayskull"
    elif ttnn.device.is_wormhole_b0(device):
        device_name = "wormhole_b0"
    yield device, device_name

    ttnn.close_device(device)


def run_conv2d_full_sweep(
    input_specs,
    input_channels,
    output_channels,
    transpose_mcast,
    output_layout,
    has_bias,
    enable_act_double_buffer,
    enable_split_reader,
    enable_subblock_padding,
    activations_dtype,
    weights_dtype,
    math_fidelity,
    fp32_accum,
    packer_l1_acc,
    groups,
    override_sharding_config,
    core_grid,
    use_shallow_conv_variant,
    deallocate_activation,
    enable_auto_formatting,
    device,
    padded_input_channels=None,
) -> list:
    [
        batch_size,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation,
    ] = input_specs
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels // groups, kernel_height, kernel_width]
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()

    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()

    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() if has_bias else None
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation, dilation),
        groups=groups,
    )

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    tt_bias_tensor = None
    if has_bias:
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=None,
        deallocate_activation=deallocate_activation,
        override_sharding_config=override_sharding_config,
        output_layout=output_layout,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_split_reader=enable_split_reader,
        enable_subblock_padding=enable_subblock_padding,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )
    if override_sharding_config:
        if len(core_grid) == 2:
            conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core_grid[0], core_grid[1])})
        elif len(core_grid) == 4:
            conv_config.core_grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(core_grid[0], core_grid[1]), ttnn.CoreRange(core_grid[2], core_grid[3])}
            )
    start_time = start_measuring_time()
    [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(kernel_height, kernel_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation, dilation),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=groups,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    return [check_with_pcc(torch_output_tensor, torch_out_golden_tensor, pcc=0.998), e2e_perf]


def run_conv2d_short_sweep(
    input_specs,
    device,
) -> list:
    # for tt-forge suite, extra arguments are tensor configs
    is_forge_suite = False
    if len(input_specs) > 15:
        [
            batch_size,
            output_channels,
            input_channels,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            groups,
            dilation_h,
            dilation_w,
            has_bias,
            [input_layout, input_buffer_type, input_dtype],
            [weights_layout, weights_buffer_type, weights_dtype],
            [output_layout, output_buffer_type, output_dtype],
        ] = input_specs
        is_forge_suite = True
    else:
        [
            batch_size,
            output_channels,
            input_channels,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            groups,
            dilation_h,
            dilation_w,
            has_bias,
        ] = input_specs
    print(input_specs)

    if is_forge_suite:
        torch_input_dtype = torch.bfloat16 if input_dtype == ttnn.DataType(ttnn.bfloat16) else torch.float32
        torch_weight_dtype = torch.bfloat16 if weights_dtype == ttnn.DataType(ttnn.bfloat16) else torch.float32

    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels // groups, kernel_height, kernel_width]
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_input_tensor_nchw = torch.randn(
        conv_input_shape, dtype=torch_input_dtype if is_forge_suite else torch.bfloat16
    ).float()

    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(
        conv_weight_shape, dtype=torch_weight_dtype if is_forge_suite else torch.bfloat16
    ).float()

    torch_bias_tensor = None
    if has_bias:
        torch_bias_tensor = (
            torch.randn(conv_bias_shape, dtype=torch_weight_dtype if is_forge_suite else torch.bfloat16).float()
            if has_bias
            else None
        )
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
        groups=groups,
    )

    tt_bias_tensor = None
    if is_forge_suite:
        input_layout = ttnn.Layout(input_layout)
        input_dtype = ttnn.DataType(input_dtype)
        input_memory_config = ttnn.DRAM_MEMORY_CONFIG if input_buffer_type == "dram" else ttnn.L1_MEMORY_CONFIG
        tt_input_tensor = ttnn.from_torch(
            torch_input_tensor, dtype=input_dtype, layout=input_layout, device=device, memory_config=input_memory_config
        )
        weights_dtype = ttnn.DataType(weights_dtype)
        tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, weights_dtype)
        if has_bias:
            tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, weights_dtype)
        output_layout = ttnn.Layout(output_layout)
        output_dtype = ttnn.DataType(output_dtype)
        conv_config = ttnn.Conv2dConfig(
            dtype=output_dtype,
            weights_dtype=weights_dtype,
            output_layout=output_layout,
        )
    else:
        tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, ttnn.bfloat16)
        if has_bias:
            tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.bfloat16)

        tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, device=device)
        conv_config = ttnn.Conv2dConfig()

    start_time = start_measuring_time()
    [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(kernel_height, kernel_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        groups=groups,
        conv_config=conv_config,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    print("End of test case")
    return [check_with_pcc(torch_output_tensor, torch_out_golden_tensor, pcc=0.998), e2e_perf]


def run_conv1d_short_sweep(
    input_specs,
    device,
) -> list:
    [
        batch_size,
        output_channels,
        input_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        groups,
        has_bias,
        dilation,
    ] = input_specs
    print(input_specs)

    # has_bias = False
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_length]
    conv_weight_shape = [output_channels, input_channels // groups, kernel_size]
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_input_tensor_ncl = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor = torch.permute(torch_input_tensor_ncl, (0, 2, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() if has_bias else None
    torch_out_golden_tensor = torch.nn.functional.conv1d(
        torch_input_tensor_ncl,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=stride,
        padding=padding,
        groups=groups,
    )

    tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, ttnn.bfloat16)
    tt_bias_tensor = None
    if has_bias:
        tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.bfloat16)

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, device=device)

    start_time = start_measuring_time()
    [tt_output_tensor_on_device, out_length, [weights_device, bias_device]] = ttnn.Conv1d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        batch_size=batch_size,
        input_length=input_length,
        groups=groups,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # torch_output_tensor is in row major layout and NLC shape
    # NLC to NCL
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_length, output_channels)

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 2, 1))

    return [check_with_pcc(torch_output_tensor, torch_out_golden_tensor, pcc=0.998), e2e_perf]
