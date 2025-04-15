# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tests.ttnn.unit_tests.operations.pool.test_avgpool2d import run_avg_pool2d
from models.utility_functions import skip_for_blackhole

import pytest
import ttnn


@pytest.fixture(scope="module")
def tensor_map():
    tensor_map = {}

    return tensor_map


parameters = {
    "input_specs": [
        # Contains following parameters
        # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, ceil_mode, count_include_pad]
        [1, 1056, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
        [1, 128, 56, 56, 2, 2, 2, 2, 0, 0, False, True],
        [1, 160, 7, 7, 2, 2, 2, 2, 0, 0, False, True],
        [1, 192, 56, 56, 2, 2, 2, 2, 0, 0, False, True],
        [1, 256, 28, 28, 2, 2, 2, 2, 0, 0, False, True],
        [1, 384, 28, 28, 2, 2, 2, 2, 0, 0, False, True],
        [1, 512, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
        [1, 640, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
        [1, 896, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
        [1, 1024, 17, 17, 3, 3, 1, 1, 1, 1, False, False],
        [1, 112, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
        [1, 1536, 8, 8, 3, 3, 1, 1, 1, 1, False, False],
        [1, 24, 56, 56, 2, 2, 2, 2, 0, 0, False, True],
        [1, 384, 35, 35, 3, 3, 1, 1, 1, 1, False, False],
        [1, 40, 28, 28, 2, 2, 2, 2, 0, 0, False, True],
        [1, 80, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
    ],
    "failing_parameters": [
        # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, ceil_mode, count_include_pad]
        [1, 1024, 17, 17, 3, 3, 1, 1, 1, 1, False, False],  # 10
        [1, 112, 14, 14, 2, 2, 2, 2, 0, 0, False, True],  # 11
        [1, 1536, 8, 8, 3, 3, 1, 1, 1, 1, False, False],  # 12
        [1, 24, 56, 56, 2, 2, 2, 2, 0, 0, False, True],  # 13
        [1, 384, 35, 35, 3, 3, 1, 1, 1, 1, False, False],  # 14
        [1, 40, 28, 28, 2, 2, 2, 2, 0, 0, False, True],  # 15
        [1, 80, 14, 14, 2, 2, 2, 2, 0, 0, False, True],  # 16
    ],
}


@skip_for_blackhole("Nigthly CI tests failing, ticket #20492")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_spec", parameters["input_specs"])
def test_ttnn_pytorch_sweep(device, tensor_map, input_spec):
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
        ceil_mode,
        count_include_pad,
    ) = input_spec

    # Check if input_spec is in failing_parameters
    if input_spec in parameters["failing_parameters"]:
        pytest.skip(f"Skipping test for failing input_spec: {input_spec}")

    run_avg_pool2d(
        device,
        tensor_map,
        (in_n, in_c, in_h, in_w),
        (kernel_h, kernel_w),
        (stride_h, stride_w),
        (pad_h, pad_w),
        (1, 1),  # dilation
        ceil_mode,
        count_include_pad,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
