# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from tests.ttnn.nightly.unit_tests.operations.pool.test_maxpool2d import run_max_pool2d
from tests.sweep_framework.sweeps.pool2d.short.max_pool2d_short_sweep import parameters


import pytest
import ttnn


# Cache map used for torch tensor reuse - the tensor will not be generated if a tensor of the same dimensions has already been generated
@pytest.fixture(scope="module")
def tensor_map(request):
    tensor_map = {}

    return tensor_map


@pytest.mark.parametrize("input_spec", parameters["max_pool2d_short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["max_pool2d_short_sweep_suite"]["dtype"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_ttnn_pytorch_sweep(device, dtype, input_spec, tensor_map):
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

    run_max_pool2d(
        [in_n, in_c, in_h, in_w],
        (kernel_h, kernel_w),
        (pad_h, pad_w),
        (stride_h, stride_w),
        (dilation_h, dilation_w),
        device,
        tensor_map,
        dtype,
        shard_scheme=None,
        ceil_mode=ceil_mode,
        nightly_skips=False,
    )
