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


def get_sliding_window_specs(
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


# Create test vectors for following 6 Params
# Batches [1, 2, 3, 4, 5, 7, 8, 16, 20, 23, 25, 27, 29, 32]
# Activations [4, 7, 16, 23, 32, 56, 60, 66, 115, 1056]
# Kernels [1, 2, 3, 4, 5, 6, 7]
# Strides [1, 2, 3, 4, 5]
# Padding [0, 1, 2, 3, 4]
# Dilation [1, 2]

# Keeping rest of the params constant as they do not affect sliding window calculations.
# Input Channels [32]
# Output Channels [32]
# Activation Datatype [ttnn.bfloat16]
# Weight Datatypes [ttnn.bfloat16]
# Groups [1]
# Transpose_mcast[True]
# use_1d_systolic_array[True]
# output_layout[ttnn.TILE_LAYOUT]

# Total test cases = 13 * 10 * 7 * 5 * 5 * 2 * 1 * 1 * 1 * 1 * 1 * 1 = 45500
# Total 7(7000*6 + 3500*1) suites are created since currently, only 10K test vectors can be generated per suite.
# Note that some test cases might be invalid.

Batches = [[1, 2], [3, 4], [5, 7], [8, 16], [20, 23], [25, 27], [32]]

parameters = {
    f"slidng_window_suite_{idx}": {
        "sliding_window_specs": list(
            get_sliding_window_specs(
                batches,  # batch_sizes
                [4, 7, 11, 16, 23, 56, 60, 66, 115, 1056],  # activation size
                [x for x in range(1, 8, 1)],  # kernel sizes
                [x for x in range(1, 6, 1)],  # stride sizes
                [x for x in range(0, 5, 1)],  # padding sizes
                [1, 2],  # dilations sizes
            )
        ),
        "input_channels": [32],
        "output_channels": [32],
        "activations_dtype": [ttnn.bfloat16],
        "weights_dtype": [ttnn.bfloat16],
        "groups": [1],
        "transpose_mcast": [True],
        "use_1d_systolic_array": [True],
        "output_layout": [ttnn.TILE_LAYOUT],
        "has_bias": [True],
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
    for idx, batches in enumerate(Batches)
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

    device_name = "Unknown"
    if ttnn.is_grayskull(device):
        device_name = "grayskull"
    elif ttnn.is_wormhole_b0(device):
        device_name = "wormhole_b0"

    yield device, device_name

    ttnn.synchronize_device(device)
    ttnn.close_device(device)


def run(
    sliding_window_specs,
    input_channels,
    output_channels,
    activations_dtype,
    weights_dtype,
    groups,
    transpose_mcast,
    use_1d_systolic_array,
    output_layout,
    has_bias,
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
    ] = sliding_window_specs
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
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
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
