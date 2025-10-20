# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from tests.sweep_framework.sweep_utils.max_pool2d_with_indices_common import run_max_pool2d_with_indices

import pytest
import ttnn


@pytest.mark.parametrize(
    "input_spec",
    [
        # Contains following parameters
        # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
        [1, 64, 112, 112, 3, 3, 2, 2, 0, 0, 1, 1, True],
        [1, 64, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, False],
        [1, 64, 147, 147, 3, 3, 2, 2, 0, 0, 1, 1, False],
        [1, 96, 56, 56, 3, 3, 2, 2, 0, 0, 1, 1, True],
        [1, 96, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, False],
        [1, 192, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, True],
        [1, 192, 56, 56, 3, 3, 2, 2, 0, 0, 1, 1, True],
        [1, 192, 71, 71, 3, 3, 2, 2, 0, 0, 1, 1, False],
        [1, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, True],
        [1, 256, 56, 56, 3, 3, 2, 2, 0, 0, 1, 1, True],
        [1, 384, 35, 35, 3, 3, 2, 2, 0, 0, 1, 1, False],
        [1, 480, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, True],
        [1, 480, 28, 28, 3, 3, 2, 2, 0, 0, 1, 1, True],
        [1, 512, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, True],
        [1, 512, 19, 19, 3, 3, 1, 1, 1, 1, 1, 1, False],
        [1, 512, 28, 28, 3, 3, 2, 2, 0, 0, 1, 1, True],
        [4, 64, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, False],
        [1, 4, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 16, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 32, 112, 112, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 64, 24, 24, 2, 2, 1, 1, 0, 0, 1, 1, False],
        [1, 64, 56, 56, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 64, 112, 112, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 64, 128, 128, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 128, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 128, 56, 56, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 128, 64, 64, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 128, 112, 112, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 128, 150, 150, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 256, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 256, 32, 32, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 256, 56, 56, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 256, 75, 75, 2, 2, 2, 2, 0, 0, 1, 1, True],
        [1, 320, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1, False],
        [1, 512, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1, False],
        # FAILING CASES
        # [1, 512, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False], # BFLOAT8 shard height must match physical height for width sharding
        # [1, 640, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False], # BFLOAT8 shard height must match physical height for width sharding
        # [1, 832, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, True], # BFLOAT8 shard height must match physical height for width sharding
        # [1, 768, 14, 14, 3, 3, 2, 2, 0, 0, 1, 1, True], # BFLOAT8 shard height must match physical height for width sharding
        # [1, 832, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, True], # BFLOAT8 shard height must match physical height for width sharding
        # [1, 1024, 17, 17, 3, 3, 2, 2, 0, 0, 1, 1, False], # BFLOAT8 shard height must match physical height for width sharding
        # [1, 32, 256, 256, 2, 2, 2, 2, 0, 0, 1, 1, False], # input size exceeds uint16 limit
        # [1, 64, 224, 224, 2, 2, 2, 2, 0, 0, 1, 1, False], # assertion errors
        # [1, 64, 300, 300, 2, 2, 2, 2, 0, 0, 1, 1, False], # input size exceeds uint16 limit
        # [1, 64, 360, 640, 3, 3, 2, 2, 1, 1, 1, 1, False], # OOM
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_with_indices_sweep(device, dtype, input_spec):
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
    ) = input_spec

    # All included test cases are verified to work with AUTO SHARDING
    # Based on comprehensive testing of GitHub issue traces using auto sharding
    # All validation (output, PCC, indices) is handled inside run_max_pool2d_with_indices function
    run_max_pool2d_with_indices(
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
        None,  # None means auto sharding
        ceil_mode,
    )

    # If we reach here, all assertions in run_max_pool2d_with_indices passed successfully
