# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.ttnn.nightly.unit_tests.operations.pool.test_maxpool2d import run_max_pool2d

# Total test cases
#   max_pool2d_full_sweep_suite_large_dims = 17 * 4 * 4 * 3 * 4 * 2 = 6528
# There can be invalid test cases in here based on conditions in invalidate_vector.

parameters = {
    "max_pool2d_full_sweep_suite_large_dims": {
        "kernel_size": [[j for i in range(2)] for j in range(15, 32)],  # square kernels only
        "padding": [[7, 7], [8, 8], [15, 15], [16, 16]],
        "stride": [[7, 7], [8, 8], [15, 15], [16, 16]],
        "sharding": [
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ],
        "shape": [
            [4, 16, 1056, 160],
            [1, 32, 599, 503],  # prime number in height and width
            [7, 31, 512, 512],  # prime numbers in batch size and channels
            [3, 17, 503, 503],  # prime numbers for all
        ],
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    }
}


def run(
    kernel_size,
    padding,
    stride,
    sharding,
    shape,
    dtype,
    *,
    device,
):
    [in_n, in_c, in_h, in_w] = shape
    [kernel_h, kernel_w] = kernel_size
    [stride_h, stride_w] = stride
    [pad_h, pad_w] = padding
    [dilation_h, dilation_w] = [1, 1]  # dilation is fix

    run_max_pool2d(
        [in_n, in_c, in_h, in_w],
        (kernel_h, kernel_w),
        (pad_h, pad_w),
        (stride_h, stride_w),
        (dilation_h, dilation_w),
        device,
        tensor_map,
        dtype,
        shard_scheme=sharding,
        ceil_mode=False,
        nightly_skips=False,
    )
