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
from tests.sweep_framework.sweep_utils.max_pool2d_common import run_max_pool2d, mesh_device_fixture


parameters = {
    "max_pool2d_short_sweep_suite": {
        "dtype": [ttnn.bfloat16],
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
            [1, 24, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False],  # requires padding along C
            [1, 480, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 480, 28, 28, 3, 3, 2, 2, 0, 0, 1, 1, True],
            [1, 512, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 512, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 512, 19, 19, 3, 3, 1, 1, 1, 1, 1, 1, False],
            [1, 512, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 512, 38, 38, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 24, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, True],  # required padding along C
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
    sharding = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
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


import pytest


@pytest.mark.parametrize("input_spec", parameters["max_pool2d_short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["max_pool2d_short_sweep_suite"]["dtype"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_localrun(device, dtype, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
    )
