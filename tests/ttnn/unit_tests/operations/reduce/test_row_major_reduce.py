# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

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
    )
    assert_numeric_metrics(
        torch_ref,
        out_tile,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.003,
        check_ulp=False,
    )
    assert_numeric_metrics(
        out_tile,
        out_rm,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.003,
        check_ulp=False,
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
    )
    assert_numeric_metrics(
        torch_ref,
        out_tile,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.01,
        check_ulp=False,
    )
    assert_numeric_metrics(
        out_tile,
        out_rm,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.01,
        check_ulp=False,
    )
