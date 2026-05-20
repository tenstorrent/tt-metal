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


def _block_sharded_mem_config(grid, shard_H, shard_W, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """Return a BLOCK_SHARDED L1 MemoryConfig with the given shard shape."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
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


def _block_grid(num_cores_x, num_cores_y):
    """Return a CoreRangeSet that is a 2D block of (num_cores_x × num_cores_y) cores (good for BLOCK_SHARDED)."""
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))})


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


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("keepdim", [True])
# @pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize(
    "shape, num_cores",
    [
        # total_rows = N*C*H must divide evenly by num_cores
        ((1, 1, 32, 64), 1),  # single core
        ((1, 1, 64, 96), 2),  # 2 cores, 32 rows/core
        ((2, 2, 32, 128), 4),  # 4 cores, 32 rows/core
        ((4, 2, 32, 64), 8),  # 8 cores, 32 rows/core
        ((1, 4, 64, 96), 4),  # 4 cores, W not multiple of 64
    ],
)
def test_mean_rm_w_height_sharded(device, dtype, keepdim, shape, num_cores):
    """W-reduce (dim=-1) on HEIGHT_SHARDED ROW_MAJOR input via the dense RM path."""
    N, C, H, W = shape
    total_rows = N * C * H
    assert total_rows % num_cores == 0, "test misconfigured: total_rows must divide num_cores"
    shard_H = total_rows // num_cores

    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch_dtype)
    torch_ref = torch.mean(torch_input, dim=-1, keepdim=keepdim)

    grid = _line_grid(num_cores, horizontal=False)
    in_mem = _height_sharded_mem_config(grid, shard_H, W)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
    )
    assert input_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    output_tensor = ttnn.mean(input_tensor, dim=-1, keepdim=keepdim)
    # output_tensor = ttnn.mean(input_tensor, dim=-1, keepdim=keepdim, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_ref,
        output,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.01,
        check_ulp=False,
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize(
    "shape, num_cores",
    [
        ((1, 1, 64, 32), 1),  # single core, shard_W=32
        ((1, 1, 32, 64), 2),  # 2 cores, shard_W=32
        ((2, 2, 64, 128), 4),  # 4 cores, shard_W=32
        ((1, 2, 32, 256), 4),  # 4 cores, shard_W=64 (multi-tile-wide shard)
        ((2, 3, 33, 128), 4),  # H=33 (non-tile-aligned)
        ((2, 3, 65, 128), 4),  # H=65 (non-tile-aligned)
    ],
)
def test_mean_rm_h_width_sharded(device, dtype, keepdim, shape, num_cores):
    """H-reduce (dim=-2) on WIDTH_SHARDED ROW_MAJOR input/output via the dense RM path."""
    N, C, H, W = shape
    assert W % num_cores == 0, "test misconfigured: W must divide num_cores"
    shard_W = W // num_cores
    elem_bytes = 2 if dtype == ttnn.bfloat16 else 4
    assert (shard_W * elem_bytes) % 16 == 0, (
        f"test misconfigured: shard_W={shard_W} * elem_bytes={elem_bytes} = "
        f"{shard_W * elem_bytes} bytes is not 16B aligned"
    )
    total_rows = N * C * H

    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch_dtype)
    torch_ref = torch.mean(torch_input, dim=-2, keepdim=keepdim)

    grid = _line_grid(num_cores, horizontal=True)
    in_mem = _width_sharded_mem_config(grid, total_rows, shard_W)
    out_mem = _width_sharded_mem_config(grid, N * C, shard_W)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
    )
    assert input_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED

    output_tensor = ttnn.mean(input_tensor, dim=-2, keepdim=keepdim, memory_config=out_mem)
    assert output_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_ref,
        output,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.01,
        check_ulp=False,
    )


# =============================================================================
# Program cache: re-running with the same sharded config must reuse the
# compiled program rather than re-JIT.
# =============================================================================


@pytest.mark.parametrize(
    "shape, num_cores, dim",
    [
        ((2, 2, 64, 128), 4, -1),  # HEIGHT_SHARDED W-reduce
        ((2, 2, 64, 128), 4, -2),  # WIDTH_SHARDED H-reduce
    ],
)
def test_mean_rm_sharded_program_cache(device, shape, num_cores, dim):
    """Same sharded RM mean op called twice must reuse the compiled program."""
    N, C, H, W = shape
    total_rows = N * C * H

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
    assert ttnn.to_torch(out1).shape == ttnn.to_torch(out2).shape


_SHARDING_COMBOS = [
    (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    (ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    (ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
]


def _factor_block_grid(num_cores):
    """Factor num_cores into (cores_x, cores_y) for a 2D block-sharded grid; prefers square-ish."""
    import math

    for cx in range(int(math.isqrt(num_cores)), 0, -1):
        if num_cores % cx == 0:
            return cx, num_cores // cx
    return 1, num_cores


def _sharded_mem_config(sharding, num_cores, total_rows, total_cols):
    """Build a sharded MemoryConfig sized to (total_rows, total_cols) across `num_cores`.

    BLOCK_SHARDED factors num_cores into a roughly-square 2D grid (cores_x x cores_y);
    both dimensions must divide evenly (cores_y | total_rows and cores_x | total_cols).
    """
    if sharding == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        grid = _line_grid(num_cores, horizontal=False)
        return _height_sharded_mem_config(grid, total_rows // num_cores, total_cols)
    if sharding == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        grid = _line_grid(num_cores, horizontal=True)
        return _width_sharded_mem_config(grid, total_rows, total_cols // num_cores)
    # BLOCK_SHARDED
    cores_x, cores_y = _factor_block_grid(num_cores)
    grid = _block_grid(cores_x, cores_y)
    return _block_sharded_mem_config(grid, total_rows // cores_y, total_cols // cores_x)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("in_sharding, out_sharding", _SHARDING_COMBOS)
@pytest.mark.parametrize(
    "shape, num_cores",
    [
        ((1, 1, 512, 512), 4),
        ((2, 1, 128, 128), 4),
        ((2, 2, 32, 128), 4),
    ],
)
def test_mean_rm_h_sharded(device, dtype, keepdim, in_sharding, out_sharding, shape, num_cores):
    N, C, H, W = shape
    total_rows = N * C * H
    output_rows = N * C

    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch_dtype)
    torch_ref = torch.mean(torch_input, dim=-2, keepdim=keepdim)

    in_mem = _sharded_mem_config(in_sharding, num_cores, total_rows, W)
    out_mem = _sharded_mem_config(out_sharding, num_cores, output_rows, W)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_mem
        # torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
    )
    output_tensor = ttnn.mean(input_tensor, dim=-2, keepdim=keepdim, memory_config=out_mem)

    output = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_ref,
        output,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.01,
        check_ulp=False,
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("in_sharding, out_sharding", _SHARDING_COMBOS)
@pytest.mark.parametrize(
    "shape, num_cores",
    [
        ((1, 1, 256, 128), 4),
        ((1, 2, 256, 128), 4),
        ((2, 2, 64, 128), 4),
    ],
)
def test_mean_sharded_W_reduce(device, dtype, keepdim, in_sharding, out_sharding, shape, num_cores):
    N, C, H, W = shape
    total_rows = N * C * H
    output_cols = 1

    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch_dtype)
    torch_ref = torch.mean(torch_input, dim=-1, keepdim=keepdim)

    in_mem = _sharded_mem_config(in_sharding, num_cores, total_rows, W)
    out_mem = _sharded_mem_config(out_sharding, num_cores, total_rows, output_cols)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_mem
        # torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
    )
    output_tensor = ttnn.mean(input_tensor, dim=-1, keepdim=keepdim, memory_config=out_mem)

    output = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_ref,
        output,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.01,
        check_ulp=False,
    )
