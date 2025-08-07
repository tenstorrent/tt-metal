# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tests.sweep_framework.sweep_utils.max_pool2d_common import run_max_pool2d
from tests.sweep_framework.sweeps.max_pool2d.short.max_pool2d_short_sweep import parameters as parameters_ttnn_pytorch
from models.utility_functions import is_wormhole_b0

from models.utility_functions import skip_for_grayskull

import pytest
import ttnn


@skip_for_grayskull()
@pytest.mark.parametrize("input_spec", parameters_ttnn_pytorch["max_pool2d_short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_ttnn_pytorch_sweep(device, dtype, input_spec):
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

    if (
        is_wormhole_b0()
        and device.core_grid.y < 8
        and in_n == 1
        and in_c == 64
        and in_h == 360
        and in_w == 640
        and dtype == ttnn.bfloat16
    ):
        pytest.skip("OOM (N300)")

    run_max_pool2d(
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
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode,
    )
