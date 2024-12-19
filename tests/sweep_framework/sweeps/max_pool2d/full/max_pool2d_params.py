# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.sweep_framework.sweep_utils.max_pool2d_common import run_max_pool2d, mesh_device_fixture, invalidate_vector

# Shapes are taken from existing unit tests
input_shapes = [
    [[1, 256, 56, 56]],
    [[1, 512, 10, 10]],
    [[2, 32, 23, 23]],
    [[4, 16, 1056, 160]],
    [[8, 4096, 10, 16]],
    [[16, 16, 528, 80]],
]

# Total test cases
#   max_pool2d_full_sweep_suite_params_{idx} = 13 * 7 * 7 * 3 * 6(input_shapes) * 2 * 2 = 45864
# There can be invalid test cases in here based on conditions in invalidate_vector.

parameters = {
    f"max_pool2d_full_sweep_suite_params_{idx}": {
        "kernel_size": [[j for i in range(2)] for j in range(2, 15)],  # square kernels only
        "padding": [[j for i in range(2)] for j in range(1, 8)],
        "stride": [[j for i in range(2)] for j in range(1, 8)],
        "sharding": [
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ],
        "shape": shape_,
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "ceil_mode": [True, False],
    }
    for idx, shape_ in enumerate(input_shapes)
}


def run(
    kernel_size,
    padding,
    stride,
    sharding,
    shape,
    dtype,
    ceil_mode=False,
    *,
    device,
):
    [in_n, in_c, in_h, in_w] = shape
    [kernel_h, kernel_w] = kernel_size
    [stride_h, stride_w] = stride
    [pad_h, pad_w] = padding
    [dilation_h, dilation_w] = [1, 1]  # dilation is fix

    return run_max_pool2d(
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
        dtype,
        device,
        sharding,
        ceil_mode,
    )
