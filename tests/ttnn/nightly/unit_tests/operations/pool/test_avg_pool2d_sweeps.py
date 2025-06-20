# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tests.ttnn.nightly.unit_tests.operations.pool.test_avgpool2d import run_avg_pool2d

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
        # [1, 48, 2, 2, 2, 2, 2, 2, 0, 0, False, True],
        # [1, 128, 56, 56, 2, 2, 2, 2, 0, 0, False, True],
        # [1, 160, 7, 7, 2, 2, 2, 2, 0, 0, False, True],
        # [1, 192, 56, 56, 2, 2, 2, 2, 0, 0, False, True],
        # [1, 256, 28, 28, 2, 2, 2, 2, 0, 0, False, True],
        # [1, 384, 28, 28, 2, 2, 2, 2, 0, 0, False, True],
        # [1, 512, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
        # [1, 640, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
        # [1, 896, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
        # [1, 1024, 17, 17, 3, 3, 1, 1, 1, 1, False, False],
        # [1, 112, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
        # [1, 1536, 8, 8, 3, 3, 1, 1, 1, 1, False, False],
        # [1, 24, 56, 56, 2, 2, 2, 2, 0, 0, False, True],
        # [1, 384, 35, 35, 3, 3, 1, 1, 1, 1, False, False],
        [1, 48, 4, 4, 2, 2, 2, 2, 0, 0, False, True],
        # [1, 80, 14, 14, 2, 2, 2, 2, 0, 0, False, True],
    ],
    "failing_parameters": [
        # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, ceil_mode, count_include_pad]
        # [1, 112, 14, 14, 2, 2, 2, 2, 0, 0, False, True],  # 11
        # [1, 24, 56, 56, 2, 2, 2, 2, 0, 0, False, True],  # 13
        # [1, 40, 28, 28, 2, 2, 2, 2, 0, 0, False, True],  # 15
        # [1, 80, 14, 14, 2, 2, 2, 2, 0, 0, False, True],  # 16
    ],
}


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
        device=device,
        tensor_map=tensor_map,
        input_shape=(in_n, in_c, in_h, in_w),
        kernel_size=(kernel_h, kernel_w),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        ceil_mode=ceil_mode,
        divisor_override=None,
        count_include_pad=count_include_pad,
        shard_scheme=None,
    )
