# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import os
import itertools
import random
import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30


def get_height_sharded_specs(
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


# Create test vectors for following 9 Params
# Kernels [2, 3, 5, 7]
# Input Channels [[3, 4, 16], [17, 25, 33], [40, 48, 64], [80, 96, 100], [160, 512, 640]]
# Output Channels [3, 4, 16, 40, 64, 96, 100, 150, 512, 1024]
# Transpose_mcast
# output_layout
# has_bias
# enable_act_double_buffer
# enable_split_reader
# enable_subblock_padding

# Keeping rest of the params constant as they do not affect sharding
# Batches
# Activations
# Dilation
# Activation Datatype
# Weight Datatypes
# Groups

# Total test cases = 4 * 15 * 10 * 2 * 2 * 2 * 2 * 2 * 2 = 38400
# Total 5(7680 each) suites are created since currently, only 10K test vectors can be generated per suite.
# Note that some test cases might be invalid.

input_channels = [[3, 4, 16], [17, 25, 33], [40, 48, 64], [80, 96, 100], [160, 512, 640]]

parameters = {
    f"height_sharded_suite_{idx}": {
        "height_sharded_specs": list(
            get_height_sharded_specs(
                [2],  # batch_sizes
                [32],  # activation size
                [2, 3, 5, 7],  # kernel sizes
                [1],  # stride sizes
                [1],  # padding sizes
                [1],  # dilations sizes
            )
        ),
        "input_channels": channels,
        "output_channels": [3, 4, 16, 40, 64, 96, 100, 150, 512, 1024],
        "transpose_mcast": [True, False],
        "output_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "has_bias": [True, False],
        "enable_act_double_buffer": [True, False],
        "enable_split_reader": [True, False],
        "enable_subblock_padding": [True, False],
        "activations_dtype": [ttnn.bfloat16],
        "weights_dtype": [ttnn.bfloat16],
        "groups": [1],
        "math_fidelity": [ttnn.MathFidelity.HiFi4],
        "config_override": [None],
        "use_shallow_conv_variant": [False],
        "enable_auto_formatting": [False],
        "padded_input_channels": [None],
        "fp32_accum": [False],
        "packer_l1_acc": [False],
        "deallocate_activation": [False],
        "debug": [False],
    }
    for idx, channels in enumerate(input_channels)
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None


def mesh_device_fixture():
    num_devices = ttnn.GetNumPCIeDevices()
    # As of now take device id as 0.
    device_id = 0
    assert device_id < num_devices, "CreateDevice not supported for non-mmio device"
    device = ttnn.CreateDevice(device_id=device_id, l1_small_size=32768)
    ttnn.SetDefaultDevice(device)

    ARCH_NAME = os.environ.get("ARCH_NAME", os.environ.get("TT_ARCH_NAME", "")).lower()
    device_name = "Unknown"
    if ARCH_NAME == "grayskull":
        device_name = "grayskull"
    elif ARCH_NAME == "wormhole_b0":
        device_name = "wormhole_b0"
    yield device, device_name

    ttnn.synchronize_device(device)
    ttnn.close_device(device)


def run(
    height_sharded_specs,
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
    groups,
    math_fidelity,
    config_override,
    use_shallow_conv_variant=False,
    enable_auto_formatting=False,
    padded_input_channels=None,
    fp32_accum=False,
    packer_l1_acc=False,
    deallocate_activation=False,
    debug=False,
    *,
    device,
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
    ] = height_sharded_specs
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
    output_shape_nhwc = [
        torch_out_golden_tensor.shape[0],
        torch_out_golden_tensor.shape[2],
        torch_out_golden_tensor.shape[3],
        torch_out_golden_tensor.shape[1],
    ]

    reader_patterns_cache = {}

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
        math_fidelity=math_fidelity,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=deallocate_activation,
        fp32_dest_acc_enabled=fp32_accum,
        packer_l1_accum_enabled=packer_l1_acc,
        output_layout=output_layout,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_split_reader=enable_split_reader,
        enable_subblock_padding=enable_subblock_padding,
    )

    start_time = start_measuring_time()
    [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
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
        conv_op_cache=reader_patterns_cache,
        debug=debug,
        groups=groups,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
    reader_patterns_cache.clear()

    return [check_with_pcc(torch_output_tensor, torch_out_golden_tensor, pcc=0.998), e2e_perf]
