# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

# Module-scoped device: these suites are large and per-test device open/close would dominate.
pytestmark = [pytest.mark.use_module_device]

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics
from models.common.utility_functions import torch_random


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        # Test cases from rm_reduce.py
        ((512, 1024, 1, 2), -1, False),
        ((512, 1024, 1, 2), -1, True),
        ((512, 1024, 1, 2), -2, False),
        ((512, 1024, 1, 2), -2, True),
        # Additional row-major compatible shapes
        ((1, 128, 256), -1, False),
        ((1, 128, 256), -1, True),
        ((1, 128, 256), -2, False),
        ((64, 512), -1, False),
        ((64, 512), -1, True),
        ((64, 512), 0, False),
        ((64, 512), 0, True),
        # More complex shapes
        ((32, 64, 128), -1, False),
        ((32, 64, 128), -1, True),
        ((32, 64, 128), 1, False),
        ((32, 64, 128), 1, True),
        ((8, 16, 32, 64), -1, False),
        ((8, 16, 32, 64), -1, True),
        ((8, 16, 32, 64), 2, False),
        ((8, 16, 32, 64), 2, True),
    ],
)
def test_mean_row_major(device, input_shape, dim, keepdim):
    """Test mean operation with ROW_MAJOR_LAYOUT (default when layout not specified)"""
    torch.manual_seed(0)
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim, keepdim)

    # Create tensor without specifying layout - defaults to ROW_MAJOR
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device)

    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Input should be in ROW_MAJOR_LAYOUT"

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)

    # PCC drops for some cases with custom RM
    if input_shape == (64, 512):
        pcc_threshold = 0.997
    else:
        pcc_threshold = 0.999
    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=pcc_threshold,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.003,
        check_ulp=True,
    )


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        # Test cases similar to rm_reduce.py
        ((512, 1024, 1, 2), -1, False),
        ((512, 1024, 1, 2), -1, True),
        ((512, 1024, 1, 2), -2, False),
        ((512, 1024, 1, 2), -2, True),
        # Additional row-major compatible shapes
        ((1, 128, 256), -1, False),
        ((1, 128, 256), -1, True),
        ((64, 512), -1, False),
        ((64, 512), 0, False),
        ((32, 64, 128), -1, False),
        ((32, 64, 128), 1, False),
        ((8, 16, 32, 64), -1, False),
        ((8, 16, 32, 64), 2, False),
    ],
)
def test_sum_row_major(device, input_shape, dim, keepdim):
    """Test sum operation with ROW_MAJOR_LAYOUT"""
    torch.manual_seed(0)
    torch_input_tensor = torch_random(input_shape, -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    # Create tensor without specifying layout - defaults to ROW_MAJOR
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device)

    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Input should be in ROW_MAJOR_LAYOUT"

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.510,
        atol=8.16,
        frobenius_threshold=0.003,
        check_ulp=True,
        ulp_threshold=65,
    )


@pytest.mark.parametrize(
    "input_shape",
    [
        # https://github.com/tenstorrent/tt-metal/issues/32830
        # (512, 1024, 1, 2),
        # (1, 128, 256),
        (64, 512),
        # (32, 64, 128),
        # (8, 16, 32, 64),
    ],
)
def test_sum_global_row_major(device, input_shape):
    """Test global sum (no dim specified) with ROW_MAJOR_LAYOUT"""
    torch.manual_seed(0)
    torch_input_tensor = torch_random(input_shape, -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor)

    # Create tensor without specifying layout - defaults to ROW_MAJOR
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device)

    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Input should be in ROW_MAJOR_LAYOUT"

    output_tensor = ttnn.sum(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        ((512, 1024, 1, 2), -1, False),
        ((512, 1024, 1, 2), -1, True),
        ((1, 128, 256), -1, False),
        ((64, 512), -1, False),
        ((32, 64, 128), -1, False),
        ((8, 16, 32, 64), -1, False),
    ],
)
def test_max_row_major(device, input_shape, dim, keepdim):
    """Test max operation with ROW_MAJOR_LAYOUT"""
    torch.manual_seed(0)
    torch_input_tensor = torch_random(input_shape, -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.max(torch_input_tensor, dim=dim, keepdim=keepdim)[0]

    # Create tensor without specifying layout - defaults to ROW_MAJOR
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device)

    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Input should be in ROW_MAJOR_LAYOUT"

    output_tensor = ttnn.max(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        # https://github.com/tenstorrent/tt-metal/issues/32829
        # ((512, 1024, 1, 2), -1, False),
        # ((512, 1024, 1, 2), -1, True),
        ((1, 128, 256), -1, False),
        ((64, 512), -1, False),
        ((32, 64, 128), -1, False),
        ((8, 16, 32, 64), -1, False),
    ],
)
def test_min_row_major(device, input_shape, dim, keepdim):
    """Test min operation with ROW_MAJOR_LAYOUT"""
    torch.manual_seed(0)
    torch_input_tensor = torch_random(input_shape, -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.min(torch_input_tensor, dim=dim, keepdim=keepdim)[0]

    # Create tensor without specifying layout - defaults to ROW_MAJOR
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device)

    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Input should be in ROW_MAJOR_LAYOUT"

    output_tensor = ttnn.min(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)
    print(torch.max(torch.abs(output_tensor - torch_output_tensor)))

    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((512, 1024, 1, 2), -1),
        ((1, 128, 256), -1),
        ((64, 512), -1),
        ((32, 64, 128), -1),
        ((8, 16, 32, 64), -1),
    ],
)
def test_std_row_major(device, input_shape, dim):
    """Test std operation with ROW_MAJOR_LAYOUT"""
    torch.manual_seed(0)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.std(torch_input_tensor, dim=dim, keepdim=False)

    # Create tensor without specifying layout - defaults to ROW_MAJOR
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device)

    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Input should be in ROW_MAJOR_LAYOUT"

    output_tensor = ttnn.std(input_tensor, dim=dim, keepdim=False)
    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    # Tolerances taken from test_reduction.py::test_std: an RM input is tilized and dispatched
    # to the same welford_reduce primitive as a TILE input (generic_reductions.cpp), so the TILE
    # bounds apply. check_ulp keeps its default (False) like the TILE version.
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.99,
        rtol=0.01,
        atol=0.01,
        frobenius_threshold=0.005,
    )


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((512, 1024, 1, 2), -1),
        ((1, 128, 256), -1),
        ((64, 512), -1),
        ((32, 64, 128), -1),
        ((8, 16, 32, 64), -1),
    ],
)
def test_var_row_major(device, input_shape, dim):
    """Test var operation with ROW_MAJOR_LAYOUT"""
    torch.manual_seed(0)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.var(torch_input_tensor, dim=dim, keepdim=False)

    # Create tensor without specifying layout - defaults to ROW_MAJOR
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device)

    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Input should be in ROW_MAJOR_LAYOUT"

    output_tensor = ttnn.var(input_tensor, dim=dim, keepdim=False)
    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    # Tolerances lifted from test_reduction.py::test_var: an RM input is tilized and dispatched
    # to the same welford_reduce primitive as a TILE input (generic_reductions.cpp), so the TILE
    # bounds apply. check_ulp keeps its default (False) like the sibling; bf16 outputs cannot
    # meet the previous 1e-06/1e-09 (sub-ULP) bounds.
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.99,
        rtol=0.01,
        atol=0.01,
        frobenius_threshold=0.007,
    )


@pytest.mark.parametrize(
    "input_shape, dims, keepdim",
    [
        # Multi-dimensional reductions
        ((32, 64, 128), [0, 1], False),
        ((32, 64, 128), [0, 1], True),
        ((32, 64, 128), [1, 2], False),
        ((8, 16, 32, 64), [0, 1], False),
        ((8, 16, 32, 64), [2, 3], False),
        ((8, 16, 32, 64), [1, 2, 3], False),
    ],
)
def test_mean_multi_dim_row_major(device, input_shape, dims, keepdim):
    """Test mean operation with multiple dimensions and ROW_MAJOR_LAYOUT"""
    torch.manual_seed(0)
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dims, keepdim=keepdim)

    # Create tensor without specifying layout - defaults to ROW_MAJOR
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    # assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Input should be in ROW_MAJOR_LAYOUT"

    output_tensor = ttnn.mean(input_tensor, dim=dims, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.98,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.002,
        check_ulp=True,
    )


@pytest.mark.parametrize(
    "input_shape, dims, keepdim",
    [
        # Multi-dimensional reductions
        # https://github.com/tenstorrent/tt-metal/issues/32830
        # ((32, 64, 128), [0, 1], False),
        ((32, 64, 128), [1, 2], False),
        # ((8, 16, 32, 64), [0, 1], False),
        ((8, 16, 32, 64), [2, 3], False),
    ],
)
def test_sum_multi_dim_row_major(device, input_shape, dims, keepdim):
    """Test sum operation with multiple dimensions and ROW_MAJOR_LAYOUT"""
    torch.manual_seed(0)
    torch_input_tensor = torch_random(input_shape, -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dims, keepdim=keepdim)

    # Create tensor without specifying layout - defaults to ROW_MAJOR
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device)

    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Input should be in ROW_MAJOR_LAYOUT"

    output_tensor = ttnn.sum(input_tensor, dim=dims, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.761,
        atol=32.64,
        frobenius_threshold=0.003,
    )


_MEAN_METRICS_BF16 = dict(
    pcc_threshold=0.998,  # some cases have reduced pcc
    rtol=0.008,
    atol=0.004,
    frobenius_threshold=0.01,
    check_ulp=False,
)


# FP32 cases have frobenius skipped for now
_MEAN_METRICS_FP32 = dict(
    pcc_threshold=0.999,
    rtol=1e-3,
    atol=1e-3,
    check_ulp=False,
    check_frobenius=False,
)


# Sum accumulates error proportionally to the reduced size. Inputs here are torch.rand in [0, 1],
# so the absolute scale of the result is bounded by the reduced dimension; tolerances are looser
# than mean but much tighter than the symmetric-range sum tests above.
_SUM_METRICS_BF16 = dict(
    pcc_threshold=0.999,
    rtol=0.05,
    atol=0.1,
    frobenius_threshold=0.02,
    check_ulp=False,
)


_SUM_METRICS_FP32 = dict(
    pcc_threshold=0.9999,
    rtol=1e-3,
    atol=1e-3,
    frobenius_threshold=1e-3,
    check_ulp=False,
)


_OPS = {
    "mean": (torch.mean, ttnn.mean),
    "sum": (torch.sum, ttnn.sum),
}


def _metrics(dtype, op):
    if op == "mean":
        return _MEAN_METRICS_FP32 if dtype == ttnn.float32 else _MEAN_METRICS_BF16
    return _SUM_METRICS_FP32 if dtype == ttnn.float32 else _SUM_METRICS_BF16


def _torch_dtype(ttnn_dtype):
    return torch.float32 if ttnn_dtype == ttnn.float32 else torch.bfloat16


def _golden(input_torch_bf_or_fp, op, dim, keepdim):
    """Reference reduction in float32 to reduce accumulation noise vs the device's mixed-precision path."""
    torch_fn, _ = _OPS[op]
    return torch_fn(input_torch_bf_or_fp.float(), dim=dim, keepdim=keepdim).to(input_torch_bf_or_fp.dtype)


# The W writer emits ROW_MAJOR only, so a TILE request runs the tilized path instead — skipped below.
@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "output_layout", [None, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["default", "rm", "tile"]
)
@pytest.mark.parametrize(
    "shape",
    [
        # tile-aligned W
        (1, 1, 1, 32),
        (1, 1, 1, 64),
        (2, 3, 4, 64),
        (4, 8, 16, 128),
        # larger NC*H
        (8, 8, 8, 64),
        (16, 4, 16, 96),
    ],
)
def test_rm_reduce_w_interleaved_tile_aligned(device, reduce_op, dtype, keepdim, output_layout, shape):
    """W reduce on ROW_MAJOR interleaved input, W a multiple of tile_width=32."""
    if output_layout == ttnn.TILE_LAYOUT:
        pytest.skip("TILE output clears use_rm_dense_w; runs the tilized path, not the RM one")
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=_torch_dtype(dtype))
    torch_ref = _golden(torch_input, reduce_op, dim=-1, keepdim=keepdim)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    assert tt_input.layout == ttnn.ROW_MAJOR_LAYOUT

    ttnn_op = _OPS[reduce_op][1]
    tt_output = ttnn_op(tt_input, dim=-1, keepdim=keepdim, output_layout=output_layout)
    # None keeps the dense RM path's natural ROW_MAJOR output.
    assert tt_output.layout == (output_layout or ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(tt_output)

    assert_numeric_metrics(torch_ref, output, **_metrics(dtype, reduce_op))


@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        # W < tile_width
        (1, 1, 1, 1),
        (1, 1, 1, 5),
        (2, 2, 2, 17),
        # W between 1 and 2 tiles
        (2, 3, 65, 1),  # NB: H large to exercise multi-row work-split
        (2, 3, 4, 65),
        # W just below 4 tiles
        (4, 2, 3, 127),
        # Wider, non-aligned
        (3, 3, 3, 99),
    ],
)
def test_rm_reduce_w_interleaved_non_tile_aligned(device, reduce_op, dtype, keepdim, shape):
    """W reduce on ROW_MAJOR interleaved input, W NOT a multiple of tile_width."""
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=_torch_dtype(dtype))
    torch_ref = _golden(torch_input, reduce_op, dim=-1, keepdim=keepdim)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    assert tt_input.layout == ttnn.ROW_MAJOR_LAYOUT

    ttnn_op = _OPS[reduce_op][1]
    tt_output = ttnn_op(tt_input, dim=-1, keepdim=keepdim)
    output = ttnn.to_torch(tt_output)

    assert_numeric_metrics(torch_ref, output, **_metrics(dtype, reduce_op))


@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "mem_cfg",
    [
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ],
    ids=["dram", "l1"],
)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 8, 64),
        (1, 1, 64, 33),
        (4, 2, 3, 127),
    ],
)
def test_rm_reduce_w_interleaved_memory_configs(device, reduce_op, dtype, mem_cfg, shape):
    """W reduce, sweep DRAM vs L1 for the interleaved RM input/output."""
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=_torch_dtype(dtype))
    torch_ref = _golden(torch_input, reduce_op, dim=-1, keepdim=False)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem_cfg,
    )

    ttnn_op = _OPS[reduce_op][1]
    tt_output = ttnn_op(tt_input, dim=-1, keepdim=False, memory_config=mem_cfg)
    output = ttnn.to_torch(tt_output)

    assert_numeric_metrics(torch_ref, output, **_metrics(dtype, reduce_op))


# Covers the RM H writer's TILE branch: only num_h_slices == 1 emits TILE directly.
@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "output_layout", [None, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["default", "rm", "tile"]
)
@pytest.mark.parametrize(
    "shape",
    [
        # tile-aligned H
        (1, 1, 32, 32),
        (1, 1, 32, 64),
        (2, 3, 64, 32),
        (4, 8, 128, 16),
        # Larger NC × W
        (8, 8, 64, 32),
        (16, 4, 32, 96),
    ],
)
def test_rm_reduce_h_interleaved_tile_aligned(device, reduce_op, dtype, keepdim, output_layout, shape):
    """H reduce on ROW_MAJOR interleaved input, H a multiple of tile_height=32."""
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=_torch_dtype(dtype))
    torch_ref = _golden(torch_input, reduce_op, dim=-2, keepdim=keepdim)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    assert tt_input.layout == ttnn.ROW_MAJOR_LAYOUT

    ttnn_op = _OPS[reduce_op][1]
    tt_output = ttnn_op(tt_input, dim=-2, keepdim=keepdim, output_layout=output_layout)
    # None keeps the dense RM path's natural ROW_MAJOR output.
    assert tt_output.layout == (output_layout or ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(tt_output)

    assert_numeric_metrics(torch_ref, output, **_metrics(dtype, reduce_op))


@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        # H < tile_height
        (1, 1, 1, 1),
        (1, 1, 5, 1),
        (2, 2, 17, 8),
        # H between 1 and 2 tiles
        (2, 3, 65, 4),
        # H just below 4 tiles
        (4, 2, 127, 3),
        # Wider, non-aligned H
        (3, 3, 99, 3),
        (1, 4, 48, 16),
    ],
)
def test_rm_reduce_h_interleaved_non_tile_aligned(device, reduce_op, dtype, keepdim, shape):
    """H reduce on ROW_MAJOR interleaved input, H NOT a multiple of tile_height."""
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=_torch_dtype(dtype))
    torch_ref = _golden(torch_input, reduce_op, dim=-2, keepdim=keepdim)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    assert tt_input.layout == ttnn.ROW_MAJOR_LAYOUT

    ttnn_op = _OPS[reduce_op][1]
    tt_output = ttnn_op(tt_input, dim=-2, keepdim=keepdim)
    output = ttnn.to_torch(tt_output)

    assert_numeric_metrics(torch_ref, output, **_metrics(dtype, reduce_op))


@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "mem_cfg",
    [
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ],
    ids=["dram", "l1"],
)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 64, 32),
        (1, 1, 33, 64),
        (4, 2, 127, 3),
    ],
)
def test_rm_reduce_h_interleaved_memory_configs(device, reduce_op, dtype, mem_cfg, shape):
    """H reduce, sweep DRAM vs L1 for the interleaved RM input/output."""
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=_torch_dtype(dtype))
    torch_ref = _golden(torch_input, reduce_op, dim=-2, keepdim=False)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem_cfg,
    )

    ttnn_op = _OPS[reduce_op][1]
    tt_output = ttnn_op(tt_input, dim=-2, keepdim=False, memory_config=mem_cfg)
    output = ttnn.to_torch(tt_output)

    assert_numeric_metrics(torch_ref, output, **_metrics(dtype, reduce_op))


@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize(
    "shape, dim",
    [
        ((2, 3, 4, 64), -1),
        ((2, 3, 4, 33), -1),  # non-tile-aligned W
        ((2, 3, 64, 32), -2),
        ((2, 3, 33, 32), -2),  # non-tile-aligned H
    ],
)
def test_rm_reduce_interleaved_program_cache(device, reduce_op, shape, dim):
    """Same RM interleaved op called twice with different data must hit the program cache."""
    torch.manual_seed(0)
    ttnn_op = _OPS[reduce_op][1]

    def run_once(seed_offset):
        torch_input = torch.rand(shape, dtype=torch.bfloat16) + seed_offset
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        tt_output = ttnn_op(tt_input, dim=dim, keepdim=False)
        return torch_input, ttnn.to_torch(tt_output)

    in1, out1 = run_once(0.0)
    in2, out2 = run_once(1.0)

    assert out1.shape == out2.shape

    ref1 = _golden(in1, reduce_op, dim=dim, keepdim=False)
    ref2 = _golden(in2, reduce_op, dim=dim, keepdim=False)
    metrics = _metrics(ttnn.bfloat16, reduce_op)
    assert_numeric_metrics(ref1, out1, **metrics)
    assert_numeric_metrics(ref2, out2, **metrics)


# Ht_rm >= 16 splits the H reduce into FP32 partials collapsed by a second stage.
@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize("fast_and_approximate_mode", [False, True])
@pytest.mark.parametrize(
    "output_layout", [None, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["default", "rm", "tile"]
)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 3136, 144),  # EfficientNetB0 global-pool; Wt=5, split fills the grid
        (1, 1, 785, 144),  # non-aligned H → last shard overhang (identity pad)
        (1, 1, 784, 145),  # non-aligned W → last-tile clamp
        (2, 3, 512, 40),  # NC>1 with tall H (Ht_rm=16)
    ],
)
def test_rm_reduce_h_axis_split(device, reduce_op, fast_and_approximate_mode, output_layout, shape):
    """H reduce on tall ROW_MAJOR input — exercises the multi-shard H-axis-split + combine path."""
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch.float32)
    torch_ref = _golden(torch_input, reduce_op, dim=-2, keepdim=False)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    assert tt_input.layout == ttnn.ROW_MAJOR_LAYOUT

    ttnn_op = _OPS[reduce_op][1]
    op_kwargs = {
        "dim": -2,
        "keepdim": False,
        "output_layout": output_layout,
        "fast_and_approximate_mode": fast_and_approximate_mode,
    }
    tt_output = ttnn_op(tt_input, **op_kwargs)
    # None keeps the dense RM path's natural ROW_MAJOR output.
    assert tt_output.layout == (output_layout or ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(tt_output)

    # Accurate SFPU (full fp32) vs FPU (tf32 truncation). The FPU path roughly doubles the relative
    # error at these depths; Quasar has no SFPU reduce LLKs, so it always takes the FPU bound.
    rtol = 0.004 if fast_and_approximate_mode or device.arch() == ttnn.device.Arch.QUASAR else 0.0011
    assert_numeric_metrics(
        torch_ref,
        output,
        pcc_threshold=0.999,
        rtol=rtol,
        atol=1e-3,
        frobenius_threshold=0.003,
        check_ulp=False,
    )


# Block-float formats only exist in TILE layout: an RM output would have to widen to BFLOAT16, so
# sum/mean reject the request instead of silently changing the dtype.
@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize("dim", [-1, -2])
def test_rm_reduce_row_major_output_rejected_for_block_float(device, reduce_op, dim, expect_error):
    torch_input = torch.rand((1, 1, 64, 128))
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_op = _OPS[reduce_op][1]

    # TILE output is supported and keeps the dtype.
    tt_output = ttnn_op(tt_input, dim=dim, output_layout=ttnn.TILE_LAYOUT)
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.dtype == ttnn.bfloat8_b, "TILE output must not change the dtype"

    with expect_error(RuntimeError, "block-float formats only exist in TILE layout"):
        ttnn_op(tt_input, dim=dim, output_layout=ttnn.ROW_MAJOR_LAYOUT)


# A row-major tensor whose H is padded stores each dim-1 slice H_padded rows apart, not H_logical.
# pad_to_tile builds that layout, and filling its pad with NaN turns any read of a pad row into a
# NaN in the result rather than an error that hides in the numerics.
@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 7, 7, 2048),  # resnet50 global pool: H 7 -> 32, 7 slices
        (1, 3, 3, 45),  # partial channel tile as well: W 45 -> 64
        (2, 5, 5, 128),  # batch > 1
        (1, 4, 40, 64),  # H_logical > tile_height: 40 -> 64
    ],
)
def test_rm_reduce_padded_h_slices(device, reduce_op, dim, shape):
    torch.manual_seed(0)
    # Zero-mean: summing 2048 uniform-positive values lands every result near 1024 with a spread of
    # only ~13, which is the same order as bf16's quantization step there, so PCC reads ~0.97 even
    # for bit-identical output. randn keeps the spread large enough for PCC to mean something.
    torch_input = torch.randn(shape, dtype=torch.bfloat16).float()
    torch_ref = _golden(torch_input, reduce_op, dim=dim, keepdim=True)

    tt_input = ttnn.Tensor(torch_input, ttnn.bfloat16).pad_to_tile(float("nan")).to(device)
    assert list(tt_input.padded_shape)[-2] != list(tt_input.shape)[-2], "shape has no padded-H coverage"

    output = ttnn.to_torch(_OPS[reduce_op][1](tt_input, dim=dim, keepdim=True))
    assert not output.isnan().any(), "result contains NaN, so a padding row was read"
    assert_numeric_metrics(torch_ref, output, **_metrics(ttnn.bfloat16, reduce_op))
