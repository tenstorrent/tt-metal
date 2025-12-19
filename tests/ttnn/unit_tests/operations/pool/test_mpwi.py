# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import math
import pytest

from tests.sweep_framework.sweep_utils.max_pool2d_with_indices_common import run_max_pool2d_with_indices


@pytest.mark.parametrize(
    "input_spec",
    [
        # Contains following parameters
        # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
        # DILATION / MULTI-BATCH CASES
        [1, 32, 8, 8, 8, 8, 1, 1, 0, 0, 1, 1, False],
    ],
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_mpwi_general(device, ttnn_dtype, input_spec):
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
        ttnn_dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
        memory_config=None,
        run_twice=False,
    )
