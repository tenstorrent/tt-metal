# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from tests.ttnn.nightly.unit_tests.operations.pool.test_avgpool2d import run_avg_pool2d
from tests.sweep_framework.sweeps.pool2d.short.avg_pool2d_short_sweep import parameters

import pytest
import ttnn


@pytest.fixture(scope="module")
def tensor_map():
    tensor_map = {}

    return tensor_map


@pytest.mark.parametrize("input_spec", parameters["avg_pool2d_short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["avg_pool2d_short_sweep_suite"]["dtype"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttnn_pytorch_sweep(device, tensor_map, input_spec, dtype):
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

    run_avg_pool2d(
        device=device,
        tensor_map=tensor_map,
        input_shape=[in_n, in_c, in_h, in_w],
        kernel_size=(kernel_h, kernel_w),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        ceil_mode=ceil_mode,
        divisor_override=None,
        count_include_pad=count_include_pad,
        shard_scheme=None,
        run_twice=False,
        in_dtype=dtype,
        nightly_skips=False,
    )
