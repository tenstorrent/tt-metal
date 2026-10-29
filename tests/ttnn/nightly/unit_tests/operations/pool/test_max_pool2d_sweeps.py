# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from tests.ttnn.nightly.unit_tests.operations.pool.test_maxpool2d import run_max_pool2d
from tests.sweep_framework.sweeps.pool2d.short.max_pool2d_short_sweep import parameters
from tests.ttnn.unit_tests.operations.pool.test_maxpool2d import (
    test_max_pool2d_dram_slice as _max_pool2d_dram_slice,
)


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


HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED
SliceWidth = ttnn.Op2DDRAMSliceWidth
SliceHeight = ttnn.Op2DDRAMSliceHeight


# Rows moved here unchanged from test_max_pool2d_dram_slice in tests/ttnn/unit_tests/operations/pool/test_maxpool2d.py,
# which keeps smaller shapes that take the same slicing paths. The two HS rows were listed twice there; the second
# copy of each now runs here.
@pytest.mark.parametrize(
    "input_spec",
    [
        # Same columns as dram_slice_tests in that file
        # Non-tile multiples
        [1, 90, 900, 900, 2, 2, 2, 2, 0, 0, 1, 1, False, 0, WS, SliceHeight, True],
        [1, 90, 900, 900, 2, 2, 2, 2, 0, 0, 1, 1, False, 0, WS, SliceWidth, True],
        # large kernel tests
        [1, 700, 700, 70, 7, 7, 3, 3, 0, 0, 1, 1, False, 0, WS, None, False],
        [1, 700, 70, 700, 7, 7, 3, 3, 0, 0, 1, 1, False, 0, WS, None, False],
        [1, 70, 700, 700, 7, 7, 3, 3, 0, 0, 1, 1, False, 0, HS, None, False],
        # large kernel wide tests
        [1, 20480, 500, 50, 9, 9, 4, 4, 0, 0, 1, 1, False, 0, WS, None, True],
        [1, 20480, 50, 500, 9, 9, 4, 4, 0, 0, 1, 1, False, 0, WS, None, True],
        [1, 384, 500, 500, 9, 9, 4, 4, 0, 0, 1, 1, False, 0, HS, None, True],
    ],
)
@pytest.mark.parametrize("in_specs", [[ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT], [ttnn.bfloat8_b, ttnn.TILE_LAYOUT]])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_dram_slice_large(device, in_specs, input_spec):
    _max_pool2d_dram_slice(device, in_specs, input_spec)
