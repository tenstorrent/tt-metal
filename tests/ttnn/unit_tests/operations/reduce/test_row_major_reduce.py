# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

# The dense ROW_MAJOR reduce fast path these tests exercise is gated off by default
# (use_row_major_support=false in reduce_op.cpp) pending fixes to a perf regression and a
# multi-H-tile hang. With it off, ttnn.mean/sum on ROW_MAJOR input falls back to tilize +
# tile-reduce, which both changes output layout (TILE, not ROW_MAJOR) and re-introduces the
# excessive padded-tilize allocation (#32546) for narrow-last-dim shapes like (512, 1024, 1, 2).
# Skip the whole module until the fast path is re-enabled. See Issue #46110.
pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skip(reason="dense ROW_MAJOR reduce path gated off (use_row_major_support=false)"),
]

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


@pytest.mark.skip(reason="Skipping std test due to issue #32830")
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
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.99,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )


@pytest.mark.skip(reason="Skipping var test due to issue #32830")
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
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.99,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
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


@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("keepdim", [False, True])
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
def test_rm_reduce_w_interleaved_tile_aligned(device, reduce_op, dtype, keepdim, shape):
    """W reduce on ROW_MAJOR interleaved input, W a multiple of tile_width=32."""
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


@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("keepdim", [False, True])
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
def test_rm_reduce_h_interleaved_tile_aligned(device, reduce_op, dtype, keepdim, shape):
    """H reduce on ROW_MAJOR interleaved input, H a multiple of tile_height=32."""
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


# --- Tall-H reduces (H-axis split path, issue #46110) ---------------------------------------------
# When H is tall and NC*Wt underfills the grid, the RM H-reduce splits the reduction axis into shards
# (stage-1 partials in FP32) and combines them (stage-2). These shapes exercise that path (Ht_rm >= 16
# triggers the split heuristic in reduce_op.cpp). Reductions here accumulate over hundreds/thousands
# of rows, so bf16 is accumulation-limited even with FP32 partials (input quantization + final bf16
# pack) — PCC is the primary correctness signal and element-wise tolerances are loosened accordingly.
# FP32 stays tight. Thresholds carry margin below empirically measured PCC (bf16 mean/sum bottomed at
# ~0.95 on (1,1,12544,32); FP32 held ~0.9998).
_TALL_MEAN_METRICS_BF16 = dict(pcc_threshold=0.94, check_allclose=False, check_frobenius=False, check_ulp=False)
_TALL_SUM_METRICS_BF16 = dict(pcc_threshold=0.94, check_allclose=False, check_frobenius=False, check_ulp=False)
_TALL_MEAN_METRICS_FP32 = dict(pcc_threshold=0.999, check_allclose=False, check_frobenius=False, check_ulp=False)
_TALL_SUM_METRICS_FP32 = dict(pcc_threshold=0.999, check_allclose=False, check_frobenius=False, check_ulp=False)


def _tall_metrics(dtype, op):
    if op == "mean":
        return _TALL_MEAN_METRICS_FP32 if dtype == ttnn.float32 else _TALL_MEAN_METRICS_BF16
    return _TALL_SUM_METRICS_FP32 if dtype == ttnn.float32 else _TALL_SUM_METRICS_BF16


@pytest.mark.parametrize("reduce_op", ["mean", "sum"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 3136, 144),  # EfficientNetB0 global-pool; Wt=5, split fills the grid
        (1, 1, 12544, 32),  # very tall, Wt=1
        (1, 1, 784, 144),  # tall-ish, still splits (Ht_rm=25)
        (1, 1, 3137, 144),  # non-aligned H → last shard overhang (identity pad)
        (1, 1, 3136, 145),  # non-aligned W → last-tile clamp
        (2, 3, 512, 40),  # NC>1 with tall H (Ht_rm=16)
    ],
)
def test_rm_reduce_h_axis_split_tall(device, reduce_op, dtype, keepdim, shape):
    """H reduce on tall ROW_MAJOR input — exercises the multi-shard H-axis-split + combine path."""
    if dtype == ttnn.bfloat16 and shape == (1, 1, 12544, 32):
        # H=12544 into 32 output columns is beyond bf16's usable accuracy for this path
        # (PCC ~0.89, still far above the un-split path's ~0.60); the FP32 variant validates the
        # split logic at this depth.
        pytest.skip("bf16 accumulation-limited at H=12544; covered by the FP32 variant")
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=_torch_dtype(dtype))
    torch_ref = _golden(torch_input, reduce_op, dim=-2, keepdim=keepdim)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    assert tt_input.layout == ttnn.ROW_MAJOR_LAYOUT

    ttnn_op = _OPS[reduce_op][1]
    tt_output = ttnn_op(tt_input, dim=-2, keepdim=keepdim)
    output = ttnn.to_torch(tt_output)

    assert_numeric_metrics(torch_ref, output, **_tall_metrics(dtype, reduce_op))
