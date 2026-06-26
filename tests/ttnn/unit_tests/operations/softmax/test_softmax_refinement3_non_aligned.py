# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3: Non-tile-aligned H/W (reduction-axis masking).

Tests that softmax correctly handles shapes where H or W is not a multiple
of 32. The kernel must mask padded lanes on the reduction axis before max
and exp, using the partial scaler mechanism.

Key test cases:
- W non-aligned, dim=-1 (W reduction) — padded columns must be excluded
- H non-aligned, dim=-2 (H reduction) — padded rows must be excluded
- Both non-aligned (W takes priority in alignment tagger)
- Aligned baselines still pass (no regression)
- TILE and ROW_MAJOR layouts both work
- fp32 and bf16 dtypes
- Fill implicit padding with garbage to verify masking actually works
"""

import pytest
import torch
import ttnn


# Shapes with non-aligned W (H aligned)
W_NON_ALIGNED_SHAPES = [
    (1, 1, 32, 50),  # W=50, partial=18
    (1, 1, 32, 17),  # W=17, partial=17
    (4, 8, 32, 47),  # W=47, partial=15, multi-batch
    (2, 1, 128, 100),  # W=100, partial=4, multi-tile W
]

# Shapes with non-aligned H (W aligned)
H_NON_ALIGNED_SHAPES = [
    (1, 1, 17, 64),  # H=17, partial=17
    (1, 1, 50, 128),  # H=50, partial=18
    (4, 8, 47, 256),  # H=47, partial=15, multi-batch
]

# Shapes with both non-aligned
BOTH_NON_ALIGNED_SHAPES = [
    (1, 1, 17, 50),  # H=17, W=50
    (2, 1, 100, 47),  # H=100, W=47
]

# Aligned baselines
ALIGNED_SHAPES = [
    (1, 1, 32, 64),
    (1, 1, 64, 128),
    (4, 8, 32, 256),
]


def _run_softmax_and_check(torch_input, dim, dtype, layout, device):
    """Run softmax on device and compare against torch reference."""
    expected = torch.softmax(torch_input.float(), dim=dim)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = ttnn.softmax(ttnn_input, dim=dim)
    result = ttnn.to_torch(ttnn_output)

    assert result.shape == expected.shape, f"Shape mismatch: got {result.shape}, expected {expected.shape}"

    rtol, atol = (0.02, 0.1) if dtype == ttnn.float32 else (0.05, 0.15)
    assert torch.allclose(
        result.float(), expected.float(), rtol=rtol, atol=atol
    ), f"Numerical mismatch: max_diff={(result.float() - expected.float()).abs().max().item():.6f}"


@pytest.mark.parametrize("shape", W_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_w_non_aligned_tile(device, shape, dim, dtype):
    """W non-aligned with TILE_LAYOUT — partial scaler masks padded columns."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16).float()
    if dtype == ttnn.float32:
        torch_input = torch_input.to(torch.float32)
    _run_softmax_and_check(torch_input, dim, dtype, ttnn.TILE_LAYOUT, device)


@pytest.mark.parametrize("shape", H_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_h_non_aligned_tile(device, shape, dim, dtype):
    """H non-aligned with TILE_LAYOUT — partial scaler masks padded rows."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16).float()
    if dtype == ttnn.float32:
        torch_input = torch_input.to(torch.float32)
    _run_softmax_and_check(torch_input, dim, dtype, ttnn.TILE_LAYOUT, device)


@pytest.mark.parametrize("shape", W_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_w_non_aligned_rm(device, shape, dim, dtype):
    """W non-aligned with ROW_MAJOR_LAYOUT — tilize wraps non-aligned W."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16).float()
    if dtype == ttnn.float32:
        torch_input = torch_input.to(torch.float32)
    _run_softmax_and_check(torch_input, dim, dtype, ttnn.ROW_MAJOR_LAYOUT, device)


@pytest.mark.parametrize("shape", H_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_h_non_aligned_rm(device, shape, dim, dtype):
    """H non-aligned with ROW_MAJOR_LAYOUT — tilize wraps non-aligned H."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16).float()
    if dtype == ttnn.float32:
        torch_input = torch_input.to(torch.float32)
    _run_softmax_and_check(torch_input, dim, dtype, ttnn.ROW_MAJOR_LAYOUT, device)


@pytest.mark.parametrize("shape", BOTH_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
def test_both_non_aligned_tile(device, shape, dim):
    """Both H and W non-aligned with TILE_LAYOUT."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    _run_softmax_and_check(torch_input, dim, ttnn.float32, ttnn.TILE_LAYOUT, device)


@pytest.mark.parametrize("shape", BOTH_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
def test_both_non_aligned_rm(device, shape, dim):
    """Both H and W non-aligned with ROW_MAJOR_LAYOUT."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    _run_softmax_and_check(torch_input, dim, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, device)


@pytest.mark.parametrize("shape", ALIGNED_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
def test_aligned_no_regression(device, shape, dim):
    """Aligned shapes still work (no regression from partial scaler changes)."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    _run_softmax_and_check(torch_input, dim, ttnn.float32, ttnn.TILE_LAYOUT, device)


def test_negative_values_non_aligned(device):
    """All-negative input: padded lanes (0) would contaminate max without masking.

    Without partial scaler: max(0, negative_values) = 0 (wrong).
    With partial scaler: max correctly excludes padded lanes.
    """
    torch.manual_seed(42)
    # Shape with W=50, all values negative
    torch_input = -torch.abs(torch.randn(1, 1, 32, 50, dtype=torch.float32)) - 1.0
    expected = torch.softmax(torch_input, dim=-1)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ttnn_output = ttnn.softmax(ttnn_input, dim=-1)
    result = ttnn.to_torch(ttnn_output)

    assert torch.allclose(
        result.float(), expected.float(), rtol=0.02, atol=0.1
    ), f"Negative values test failed: max_diff={(result.float() - expected.float()).abs().max().item():.6f}"


def test_garbage_padding_masked(device):
    """Fill implicit padding with garbage — partial scaler must exclude it.

    This verifies the partial scaler actually masks padded positions, not
    just relies on zero padding from from_torch.
    """
    torch.manual_seed(42)
    shape = (1, 1, 32, 50)
    torch_input = torch.randn(shape, dtype=torch.float32)
    expected = torch.softmax(torch_input, dim=-1)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    # Fill implicit padding with large garbage values
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, 99.0)

    ttnn_output = ttnn.softmax(ttnn_input, dim=-1)
    result = ttnn.to_torch(ttnn_output)

    # If masking works, the garbage padding doesn't affect the result
    assert torch.allclose(
        result.float(), expected.float(), rtol=0.02, atol=0.1
    ), f"Garbage padding test failed: max_diff={(result.float() - expected.float()).abs().max().item():.6f}"


def test_garbage_padding_masked_h_non_aligned(device):
    """H non-aligned with garbage padding — partial scaler must exclude it."""
    torch.manual_seed(42)
    shape = (1, 1, 50, 64)
    torch_input = torch.randn(shape, dtype=torch.float32)
    expected = torch.softmax(torch_input, dim=-2)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, 99.0)

    ttnn_output = ttnn.softmax(ttnn_input, dim=-2)
    result = ttnn.to_torch(ttnn_output)

    assert torch.allclose(
        result.float(), expected.float(), rtol=0.02, atol=0.1
    ), f"H garbage padding test failed: max_diff={(result.float() - expected.float()).abs().max().item():.6f}"


def test_bf8b_non_aligned_excluded(device):
    """bf8b + non-aligned should raise ExcludedCell (in EXCLUSIONS)."""
    torch_input = torch.randn(1, 1, 32, 50, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    with pytest.raises(NotImplementedError, match="unsupported combination"):
        ttnn.softmax(ttnn_input, dim=-1)
