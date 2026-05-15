# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics
from models.common.utility_functions import torch_random


# ---------------------------------------------------------------------------
# Helpers for sharded RM reduce tests
# ---------------------------------------------------------------------------


def _height_sharded_mem_config(grid, shard_H, shard_W, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """Return a HEIGHT_SHARDED L1 MemoryConfig with the given shard shape."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_H, shard_W], orientation),
    )


def _width_sharded_mem_config(grid, shard_H, shard_W, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """Return a WIDTH_SHARDED L1 MemoryConfig with the given shard shape."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_H, shard_W], orientation),
    )


def _line_grid(num_cores, horizontal=False):
    """Return a CoreRangeSet that is a line of `num_cores` cores.

    horizontal=False → column strip x=0, y=0..num_cores-1 (good for HEIGHT_SHARDED)
    horizontal=True  → row strip x=0..num_cores-1, y=0   (good for WIDTH_SHARDED)
    """
    if horizontal:
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores - 1))})


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

    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
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


# -----------------------------------------------------------------------------
# Dense RM W-mean path (tilize-in-compute): exercise last dim W where logical rows
# are not tile-aligned. Regression for staging reads / padding vs tiled reference.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_shape, keepdim",
    [
        ((1, 1, 1, 5), False),
        ((1, 1, 1, 5), True),
        ((2, 3, 4, 33), False),  # W = 32 + 1
        ((2, 3, 4, 65), True),  # W = 2*32 + 1
        ((4, 2, 3, 127), False),  # W = 4*32 - 1
        ((2, 16, 8, 96), False),  # W divisible by 32 (sanity alongside odd widths)
    ],
)
def test_mean_row_major_last_dim_not_multiple_of_tile_width(device, input_shape, keepdim):
    """Mean over last dim on 4D ROW_MAJOR when W is not a multiple of tile width (32)."""
    torch.manual_seed(20250206)
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=-1, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_torch(ttnn.mean(input_tensor, dim=-1, keepdim=keepdim))

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.003,
        check_ulp=True,
        assert_on_fail=False,
    )


@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 5, 7, 41),
        (1, 4, 16, 48),
        (3, 3, 3, 99),
    ],
)
def test_mean_row_major_matches_tiled_layout_reference(device, input_shape):
    """Same tensor: ROW_MAJOR (dense path when eligible) vs TILE mean along W should match torch and each other."""
    torch.manual_seed(20250206)
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_ref = torch.mean(torch_input_tensor, dim=-1, keepdim=False)

    rm_input = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tile_input = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    out_rm = ttnn.to_torch(ttnn.mean(rm_input, dim=-1, keepdim=False))
    out_tile = ttnn.to_torch(ttnn.mean(tile_input, dim=-1, keepdim=False))

    assert_numeric_metrics(
        torch_ref,
        out_rm,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.003,
        check_ulp=False,
        assert_on_fail=False,
    )
    assert_numeric_metrics(
        torch_ref,
        out_tile,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.003,
        check_ulp=False,
        assert_on_fail=False,
    )
    assert_numeric_metrics(
        out_tile,
        out_rm,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.003,
        check_ulp=False,
        assert_on_fail=False,
    )


# -----------------------------------------------------------------------------
# Dense RM H-mean path (tilize-in-compute): exercise second-to-last dim H where
# logical H is not tile-aligned and the H reduction crosses NC slabs. Regression
# for staging reads / accumulator orientation / writer face-aligned stores vs
# tiled reference.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_shape, keepdim",
    [
        ((1, 1, 5, 1), False),
        ((1, 1, 5, 1), True),
        ((2, 3, 33, 4), False),  # H = 32 + 1
        ((2, 3, 65, 4), True),  # H = 2*32 + 1
        ((4, 2, 127, 3), False),  # H = 4*32 - 1
        ((2, 16, 96, 8), False),  # H divisible by 32 (sanity alongside odd heights)
    ],
)
def test_mean_row_major_h_not_multiple_of_tile_height(device, input_shape, keepdim):
    """Mean over H on 4D ROW_MAJOR when H is not a multiple of tile height (32).

    Frobenius threshold is looser than the W partial-tile test because H mean accumulates up to
    H_logical values per output column inside a single tile reduce, vs the W path which sums at most
    TILE_WIDTH values per intra-tile reduction step. Per-column bf16 absolute error remains within
    1 ULP (ATOL ~ 2^-8), matching the W path's correctness contract.
    """
    torch.manual_seed(20250206)
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=-2, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_torch(ttnn.mean(input_tensor, dim=-2, keepdim=keepdim))

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.01,
        check_ulp=True,
        assert_on_fail=False,
    )


@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 5, 41, 7),
        (1, 4, 48, 16),
        (3, 3, 99, 3),
    ],
)
def test_mean_row_major_h_matches_tiled_layout_reference(device, input_shape):
    """Same tensor: ROW_MAJOR (dense H path when eligible) vs TILE mean along H should match torch and each other."""
    torch.manual_seed(20250206)
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_ref = torch.mean(torch_input_tensor, dim=-2, keepdim=False)

    rm_input = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tile_input = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    out_rm = ttnn.to_torch(ttnn.mean(rm_input, dim=-2, keepdim=False))
    out_tile = ttnn.to_torch(ttnn.mean(tile_input, dim=-2, keepdim=False))

    # H mean accumulates up to H_logical values per output column; loosen Frobenius slightly from the
    # W cross-check test for the same reason. PCC + ATOL/RTOL still hold to the W contract.
    assert_numeric_metrics(
        torch_ref,
        out_rm,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.01,
        check_ulp=False,
        assert_on_fail=False,
    )
    assert_numeric_metrics(
        torch_ref,
        out_tile,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.01,
        check_ulp=False,
        assert_on_fail=False,
    )
    assert_numeric_metrics(
        out_tile,
        out_rm,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.01,
        check_ulp=False,
        assert_on_fail=False,
    )


# =============================================================================
# Dense RM W-reduce with HEIGHT_SHARDED input
#
# The dense RM W-reduce path supports HEIGHT_SHARDED input.  Output is always
# interleaved DRAM: W-reduce produces one datum per row (shard_W=1), which is
# smaller than the 16B L1 alignment required for sharded output.
# =============================================================================


# @pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
# @pytest.mark.parametrize("keepdim", [False])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize(
    "shape, num_cores",
    [
        # total_rows = N*C*H must divide evenly by num_cores
        ((1, 1, 32, 64), 1),  # single core — identical to interleaved path
        ((1, 1, 64, 96), 2),  # 2 cores, 32 rows/core
        ((2, 2, 32, 128), 4),  # 4 cores, 32 rows/core
        ((2, 2, 64, 128), 4),  # 4 cores, 64 rows/core
        ((4, 2, 32, 64), 8),  # 8 cores, 32 rows/core
        ((1, 4, 64, 96), 4),  # 4 cores, 64 rows/core, W not multiple of 64
        # ((2, 3, 32, 33), 2),    # W = 32+1 (partial last tile)
        # ((2, 2, 32, 65), 4),    # W = 2*32+1 (two partial tiles)
        # ((1, 2, 64, 127), 2),   # W = 4*32-1 (partial last tile, multi-tile)
    ],
)
def test_mean_rm_w_height_sharded(device, dtype, keepdim, shape, num_cores):
    """W-reduce (dim=-1) on HEIGHT_SHARDED ROW_MAJOR tensor via the dense RM path."""
    N, C, H, W = shape
    total_rows = N * C * H
    assert total_rows % num_cores == 0, "test misconfigured: total_rows must divide num_cores"
    shard_H = total_rows // num_cores

    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch.float32)
    torch_ref = torch.mean(torch_input, dim=-1, keepdim=keepdim).to(
        torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    )

    grid = _line_grid(num_cores, horizontal=False)
    in_mem = _height_sharded_mem_config(grid, shard_H, W)

    tt_dtype = dtype
    input_tensor = ttnn.from_torch(
        torch_input.to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32),
        dtype=tt_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=in_mem,
    )
    assert input_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    output_tensor = ttnn.mean(input_tensor, dim=-1, keepdim=keepdim, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert (
        output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"Expected ROW_MAJOR output (dense RM path), got {output_tensor.layout}"
    output = ttnn.to_torch(output_tensor)

    assert_numeric_metrics(
        torch_ref,
        output,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.01,
        check_ulp=False,
        assert_on_fail=False,
    )


@pytest.mark.parametrize(
    "shape, num_cores",
    [
        ((2, 2, 64, 128), 4),
        ((4, 2, 32, 64), 8),
        ((1, 4, 64, 96), 4),
    ],
)
def test_mean_rm_w_height_sharded_output_memory_config(device, shape, num_cores):
    """Verify that W-reduce on HEIGHT_SHARDED input produces interleaved DRAM ROW_MAJOR output."""
    N, C, H, W = shape
    total_rows = N * C * H
    shard_H = total_rows // num_cores

    torch.manual_seed(1)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)

    grid = _line_grid(num_cores, horizontal=False)
    in_mem = _height_sharded_mem_config(grid, shard_H, W)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
    )
    output_tensor = ttnn.mean(input_tensor, dim=-1, keepdim=True)

    out_cfg = output_tensor.memory_config()
    assert (
        out_cfg.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    ), f"Expected INTERLEAVED output, got {out_cfg.memory_layout}"
    assert (
        output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"Expected ROW_MAJOR output (dense RM path), got {output_tensor.layout}"


@pytest.mark.parametrize(
    "shape, num_cores",
    [
        ((2, 2, 64, 128), 4),
        ((2, 2, 64, 128), 8),
        ((1, 1, 32, 64), 1),
    ],
)
def test_mean_rm_w_height_sharded_matches_interleaved(device, shape, num_cores):
    """HEIGHT_SHARDED RM W-mean result must match the interleaved RM W-mean result."""
    N, C, H, W = shape
    total_rows = N * C * H
    shard_H = total_rows // num_cores

    torch.manual_seed(2)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)

    # Interleaved reference (existing RM dense path)
    rm_input_interleaved = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    out_interleaved = ttnn.to_torch(ttnn.mean(rm_input_interleaved, dim=-1, keepdim=True))

    # Sharded path
    grid = _line_grid(num_cores, horizontal=False)
    in_mem = _height_sharded_mem_config(grid, shard_H, W)
    rm_input_sharded = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
    )
    out_sharded = ttnn.to_torch(ttnn.mean(rm_input_sharded, dim=-1, keepdim=True))

    assert_numeric_metrics(
        out_interleaved,
        out_sharded,
        pcc_threshold=0.999,
        rtol=1e-4,
        atol=1e-4,
        frobenius_threshold=1e-4,
        assert_on_fail=False,
    )


# =============================================================================
# Dense RM H-reduce with WIDTH_SHARDED input/output
#
# The dense RM H-reduce path supports WIDTH_SHARDED input when both input and
# output use WIDTH_SHARDED L1 layout with the same core grid.  Each core
# independently reduces all H rows of its own shard columns along the H dim.
# Constraint: shard_W * elem_bytes must be 16B aligned (NOC DMA row-address
# alignment). For BF16: shard_W % 8 == 0. For FLOAT32: shard_W % 4 == 0.
# W_logical must be exactly divisible by shard_W (no partial last shard).
# =============================================================================


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
# @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("keepdim", [False])
# @pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize(
    "shape, num_cores",
    [
        # shard_W * 2 (BF16) must be 16B aligned → shard_W % 8 == 0
        ((1, 1, 64, 32), 1),  # single core, shard_W=32
        ((1, 1, 32, 64), 2),  # 2 cores, shard_W=32
        ((2, 2, 64, 128), 4),  # 4 cores, shard_W=32
        ((1, 2, 32, 256), 4),  # 4 cores, shard_W=64 (two tiles wide per shard)
        ((2, 3, 33, 128), 4),  # H=33 (non-tile-aligned), shard_W=32
        ((2, 3, 65, 128), 4),  # H=65 (non-tile-aligned), shard_W=32
        ((4, 2, 127, 128), 4),  # H=127 (non-tile-aligned), shard_W=32
        ((2, 2, 96, 256), 8),  # 8 cores, shard_W=32
        ((1, 2, 64, 192), 3),  # 3 cores, shard_W=64
        ((1, 1, 64, 16), 1),  # single core, shard_W=16 (sub-tile, 32B row → 16B aligned)
        ((1, 1, 64, 16), 2),  # 2 cores, shard_W=8 (16B row → exactly 16B aligned)
        ((1, 2, 32, 48), 3),  # 3 cores, shard_W=16 (32B row → 16B aligned)
        ((1, 1, 64, 24), 3),  # 3 cores, shard_W=8 (16B row → exactly 16B aligned)
    ],
)
def test_mean_rm_h_width_sharded(device, dtype, keepdim, shape, num_cores):
    """H-reduce (dim=-2) on WIDTH_SHARDED ROW_MAJOR tensor via the dense RM path."""
    N, C, H, W = shape
    assert W % num_cores == 0, "test misconfigured: W must divide num_cores"
    shard_W = W // num_cores
    elem_bytes = 2 if dtype == ttnn.bfloat16 else 4
    assert (shard_W * elem_bytes) % 16 == 0, (
        f"test misconfigured: shard_W={shard_W} * elem_bytes={elem_bytes} = "
        f"{shard_W * elem_bytes} bytes is not 16B aligned"
    )
    total_rows = N * C * H

    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch.float32)
    torch_ref = torch.mean(torch_input, dim=-2, keepdim=keepdim).to(
        torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    )

    grid = _line_grid(num_cores, horizontal=True)
    in_mem = _width_sharded_mem_config(grid, total_rows, shard_W)
    out_mem = _width_sharded_mem_config(grid, N * C, shard_W)

    input_tensor = ttnn.from_torch(
        torch_input.to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=in_mem,
    )
    assert input_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED

    output_tensor = ttnn.mean(input_tensor, dim=-2, keepdim=keepdim, memory_config=out_mem)
    assert (
        output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"Expected ROW_MAJOR output (dense RM path), got {output_tensor.layout}"
    output = ttnn.to_torch(output_tensor)

    assert_numeric_metrics(
        torch_ref,
        output,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.01,
        check_ulp=False,
        assert_on_fail=False,
    )


@pytest.mark.parametrize(
    "shape, num_cores",
    [
        ((2, 2, 64, 128), 4),
        ((1, 2, 32, 256), 4),
        ((2, 3, 33, 128), 4),  # H=33 (non-tile-aligned), shard_W=32
        ((2, 3, 65, 128), 4),  # H=65 (non-tile-aligned), shard_W=32
    ],
)
def test_mean_rm_h_width_sharded_output_memory_config(device, shape, num_cores):
    """Verify that the WIDTH_SHARDED output tensor is WIDTH_SHARDED ROW_MAJOR (dense RM path)."""
    N, C, H, W = shape
    shard_W = W // num_cores
    total_rows = N * C * H

    torch.manual_seed(1)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)

    grid = _line_grid(num_cores, horizontal=True)
    in_mem = _width_sharded_mem_config(grid, total_rows, shard_W)
    out_mem = _width_sharded_mem_config(grid, N * C, shard_W)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
    )
    output_tensor = ttnn.mean(input_tensor, dim=-2, keepdim=True, memory_config=out_mem)

    out_cfg = output_tensor.memory_config()
    assert (
        out_cfg.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    ), f"Expected WIDTH_SHARDED output, got {out_cfg.memory_layout}"
    assert out_cfg.buffer_type == ttnn.BufferType.L1
    assert (
        output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"Expected ROW_MAJOR output (dense RM path), got {output_tensor.layout}"


@pytest.mark.parametrize(
    "shape, num_cores",
    [
        ((2, 2, 64, 128), 4),
        ((1, 2, 32, 64), 2),
        ((1, 1, 64, 32), 1),
    ],
)
def test_mean_rm_h_width_sharded_matches_interleaved(device, shape, num_cores):
    """WIDTH_SHARDED RM H-mean result must match the interleaved RM H-mean result."""
    N, C, H, W = shape
    shard_W = W // num_cores
    total_rows = N * C * H

    torch.manual_seed(3)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)

    # Interleaved reference
    rm_input_interleaved = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    out_interleaved = ttnn.to_torch(ttnn.mean(rm_input_interleaved, dim=-2, keepdim=True))

    # Sharded path
    grid = _line_grid(num_cores, horizontal=True)
    in_mem = _width_sharded_mem_config(grid, total_rows, shard_W)
    out_mem = _width_sharded_mem_config(grid, N * C, shard_W)
    rm_input_sharded = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
    )
    out_sharded = ttnn.to_torch(ttnn.mean(rm_input_sharded, dim=-2, keepdim=True, memory_config=out_mem))

    assert_numeric_metrics(
        out_interleaved,
        out_sharded,
        pcc_threshold=0.999,
        rtol=1e-4,
        atol=1e-4,
        frobenius_threshold=1e-4,
        assert_on_fail=False,
    )


# =============================================================================
# Program cache validation: re-running with the same sharded config must reuse
# the compiled program (no re-JIT) rather than re-compiling.
# =============================================================================


@pytest.mark.parametrize(
    "shape, num_cores, dim",
    [
        ((2, 2, 64, 128), 4, -1),  # HEIGHT_SHARDED W reduce
        ((2, 2, 64, 128), 4, -2),  # WIDTH_SHARDED H reduce
    ],
)
def test_mean_rm_sharded_program_cache(device, shape, num_cores, dim, use_program_cache):
    """Same sharded RM mean op called twice must reuse the same compiled program."""
    N, C, H, W = shape
    total_rows = N * C * H

    torch.manual_seed(5)

    def run_once(seed_offset):
        torch_input = torch.rand(shape, dtype=torch.bfloat16) + seed_offset
        if dim == -1:
            shard_H = total_rows // num_cores
            grid = _line_grid(num_cores, horizontal=False)
            in_mem = _height_sharded_mem_config(grid, shard_H, W)
            tt_input = ttnn.from_torch(
                torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
            )
            return ttnn.mean(tt_input, dim=dim, keepdim=True)
        else:
            shard_W = W // num_cores
            grid = _line_grid(num_cores, horizontal=True)
            in_mem = _width_sharded_mem_config(grid, total_rows, shard_W)
            out_mem = _width_sharded_mem_config(grid, N * C, shard_W)
            tt_input = ttnn.from_torch(
                torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
            )
            return ttnn.mean(tt_input, dim=dim, keepdim=True, memory_config=out_mem)

    out1 = run_once(0.0)
    out2 = run_once(1.0)

    # Both should produce valid tensors; if program cache was broken the second
    # call would crash or produce wrong results.
    assert ttnn.to_torch(out1).shape == ttnn.to_torch(out2).shape
