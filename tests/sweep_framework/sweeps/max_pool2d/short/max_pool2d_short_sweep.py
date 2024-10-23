# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import itertools
import random
import torch
import math

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


parameters = {
    "max_pool2d_short_sweep_suite": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            [1, 128, 112, 112, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 128, 150, 150, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 128, 56, 56, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 128, 64, 64, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 16, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 192, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 192, 56, 56, 3, 3, 2, 2, 0, 0, 1, 1, True],
            [1, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 256, 32, 32, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 56, 56, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 75, 75, 2, 2, 2, 2, 0, 0, 1, 1, True],
            [1, 32, 256, 256, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 320, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 4, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 480, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 480, 28, 28, 3, 3, 2, 2, 0, 0, 1, 1, True],
            [1, 512, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 512, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 512, 19, 19, 3, 3, 1, 1, 1, 1, 1, 1, False],
            [1, 512, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 512, 38, 38, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 528, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 64, 112, 112, 3, 3, 2, 2, 0, 0, 1, 1, True],
            [1, 64, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 64, 128, 128, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 64, 224, 224, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 64, 24, 24, 2, 2, 1, 1, 0, 0, 1, 1, False],
            [1, 64, 300, 300, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 64, 360, 640, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 64, 400, 544, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 640, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 832, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, True],
            [1, 832, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 96, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, False],
        ],
    },
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
    if device.arch() == "grayskull":
        device_name = "grayskull"
    elif device.arch() == "wormhole_b0":
        device_name = "wormhole_b0"
    yield device, device_name

    ttnn.close_device(device)


def run(
    input_specs,
    dtype,
    *,
    device,
):
    (
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_specs
    act_shape = in_n, in_c, in_h, in_w
    kernel_size = kernel_h, kernel_w
    padding = pad_h, pad_w
    stride = stride_h, stride_w
    dilation = dilation_h, dilation_w

    out_h = math.floor((in_h + 2 * pad_h - (dilation_h * kernel_h - 1) - 1) / stride_h) + 1
    out_w = math.floor((in_w + 2 * pad_w - (dilation_w * kernel_w - 1) - 1) / stride_w) + 1

    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    act = torch.randn(act_shape, dtype=torch.bfloat16)
    act_shape = (1, 1, in_n * in_h * in_w, in_c)
    act_permuted = torch.permute(act, (0, 2, 3, 1))
    act_reshaped = act_permuted.reshape(act_shape)

    if dtype == ttnn.bfloat8_b:
        ttact = ttnn.from_torch(act_reshaped, dtype, layout=ttnn.TILE_LAYOUT)
    else:
        ttact = ttnn.from_torch(act_reshaped, dtype)

    ttact_device = ttnn.to_device(ttact, device)
    parallel_config = ttnn._ttnn.operations.conv2d.determine_parallel_config(
        is_1d_systolic=True,
        batch_size=in_n,
        input_channels=in_c,
        output_height=out_h,
        output_width=out_w,
        output_channels=in_c,
        device=device,
        is_out_tiled=False,
    )
    sharded_memory_config = ttnn._ttnn.operations.conv2d.create_sharded_memory_config_from_parallel_config(
        tensor_shape=act_shape,
        parallel_config=parallel_config,
        tile_size=32 if dtype == ttnn.bfloat8_b else 1,
    )
    ttact_device = ttnn.to_memory_config(ttact_device, sharded_memory_config)
    start_time = start_measuring_time()
    output = ttnn.max_pool2d(
        input_tensor=ttact_device,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=[kernel_h, kernel_w],
        stride=[stride_h, stride_w],
        padding=[pad_h, pad_w],
        dilation=[dilation_h, dilation_w],
        device=device,
    )

    # interleaved_mem_config = ttnn.L1_MEMORY_CONFIG
    # output = ttnn.to_memory_config(output, interleaved_mem_config)
    output_host = output.cpu()
    output_pytorch_padded = ttnn.to_torch(output_host)
    output_pytorch = output_pytorch_padded[:, :, :, :in_c]
    e2e_perf = stop_measuring_time(start_time)

    ## reference
    golden_pytorch = torch.nn.MaxPool2d(
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=False,
        ceil_mode=ceil_mode,
    )(act)

    ## test for equivalance
    golden_shape = golden_pytorch.shape
    output_pytorch = output_pytorch.reshape(golden_shape[0], golden_shape[2], golden_shape[3], golden_shape[1])
    output_pytorch = torch.permute(output_pytorch, (0, 3, 1, 2))  ## N, C, H, W

    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    if dtype == ttnn.bfloat8_b:
        atol = 0.35

    allclose = torch.allclose(output_pytorch, golden_pytorch, atol=atol)
    isequal = torch.equal(output_pytorch, golden_pytorch)

    assert allclose, " Reference and output tensor are not close"
    if dtype == ttnn.bfloat16:
        assert isequal, " Reference and output tensor are not equal"

    # check pcc and return
    return [check_with_pcc(output_pytorch, golden_pytorch, pcc=0.998), e2e_perf]
