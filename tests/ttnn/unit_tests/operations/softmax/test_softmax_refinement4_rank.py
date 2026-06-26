# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 tests: rank expansion (2D, 3D tensors).

Tests that softmax correctly handles rank-2 (B, H) and rank-3 (B, S, H)
input tensors by internally unsqueezing to 4D and reshaping the output back.

Verifies:
- Output shape matches input rank (not 4D)
- Numerical correctness vs torch.softmax
- Both dim=-1 and dim=-2 work for all ranks
- TILE and ROW_MAJOR layouts work
- Non-tile-aligned shapes work (masking from Refinement 3 carries over)
- bf16 dtype works
- Positive dim aliases canonicalize correctly
- Multi-core distribution works (larger batches)
"""

from __future__ import annotations

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# Shape fixtures
# ---------------------------------------------------------------------------

RANK2_SHAPES = [
    (32, 64),  # tile-aligned, single tile pair
    (128, 512),  # tile-aligned, multi-tile
    (32, 17),  # w_non_aligned
    (17, 64),  # h_non_aligned
]

RANK3_SHAPES = [
    (1, 32, 128),  # tile-aligned
    (4, 128, 512),  # tile-aligned, multi-batch
    (1, 32, 50),  # w_non_aligned
    (1, 17, 128),  # h_non_aligned
]

RANK4_SHAPES = [
    (1, 1, 32, 64),
    (2, 4, 64, 128),
]


# ---------------------------------------------------------------------------
# Basic correctness: output shape + numerical accuracy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", RANK2_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
def test_rank2_tile_layout(device, shape, dim):
    """Rank-2 (B, H) with TILE_LAYOUT."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)
    expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.operations.softmax.softmax(ttnn_input, dim=dim)
    result = ttnn.to_torch(ttnn_output)

    # Shape must match input rank (not 4D)
    assert tuple(result.shape) == shape, f"Expected {shape}, got {tuple(result.shape)}"
    assert_with_pcc(result, expected, pcc=0.999)


@pytest.mark.parametrize("shape", RANK3_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
def test_rank3_tile_layout(device, shape, dim):
    """Rank-3 (B, S, H) with TILE_LAYOUT."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)
    expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.operations.softmax.softmax(ttnn_input, dim=dim)
    result = ttnn.to_torch(ttnn_output)

    assert tuple(result.shape) == shape, f"Expected {shape}, got {tuple(result.shape)}"
    assert_with_pcc(result, expected, pcc=0.999)


@pytest.mark.parametrize("shape", RANK2_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
def test_rank2_row_major(device, shape, dim):
    """Rank-2 (B, H) with ROW_MAJOR_LAYOUT."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)
    expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.operations.softmax.softmax(ttnn_input, dim=dim)
    result = ttnn.to_torch(ttnn_output)

    assert tuple(result.shape) == shape, f"Expected {shape}, got {tuple(result.shape)}"
    assert_with_pcc(result, expected, pcc=0.999)


@pytest.mark.parametrize("shape", RANK3_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
def test_rank3_row_major(device, shape, dim):
    """Rank-3 (B, S, H) with ROW_MAJOR_LAYOUT."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)
    expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output = ttnn.operations.softmax.softmax(ttnn_input, dim=dim)
    result = ttnn.to_torch(ttnn_output)

    assert tuple(result.shape) == shape, f"Expected {shape}, got {tuple(result.shape)}"
    assert_with_pcc(result, expected, pcc=0.999)


# ---------------------------------------------------------------------------
# dtype: bf16 for rank 2/3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 64), (128, 512), (4, 128, 512)])
@pytest.mark.parametrize("dim", [-1, -2])
def test_rank2_3_bf16(device, shape, dim):
    """bf16 dtype with rank-2 and rank-3 tensors."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.operations.softmax.softmax(ttnn_input, dim=dim)
    result = ttnn.to_torch(ttnn_output)

    assert tuple(result.shape) == shape, f"Expected {shape}, got {tuple(result.shape)}"
    assert_with_pcc(result.float(), expected.float(), pcc=0.99)


# ---------------------------------------------------------------------------
# Positive dim aliases (canonicalization)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, dim_positive, dim_negative",
    [
        ((32, 64), 1, -1),  # rank-2: dim=1 ≡ dim=-1
        ((32, 64), 0, -2),  # rank-2: dim=0 ≡ dim=-2
        ((4, 128, 512), 2, -1),  # rank-3: dim=2 ≡ dim=-1
        ((4, 128, 512), 1, -2),  # rank-3: dim=1 ≡ dim=-2
    ],
)
def test_positive_dim_alias(device, shape, dim_positive, dim_negative):
    """Positive dim aliases must produce the same result as negative dims."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # Using positive dim
    ttnn_out_pos = ttnn.operations.softmax.softmax(ttnn_input, dim=dim_positive)
    result_pos = ttnn.to_torch(ttnn_out_pos)

    # Using negative dim
    ttnn_out_neg = ttnn.operations.softmax.softmax(ttnn_input, dim=dim_negative)
    result_neg = ttnn.to_torch(ttnn_out_neg)

    # Both should match torch.softmax with the negative dim
    expected = torch.softmax(torch_input, dim=dim_negative)
    assert_with_pcc(result_pos, expected, pcc=0.999)
    assert_with_pcc(result_neg, expected, pcc=0.999)
    assert torch.allclose(result_pos, result_neg, atol=1e-5)


# ---------------------------------------------------------------------------
# Non-tile-aligned shapes with rank 2/3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, dim",
    [
        # rank-2 w_non_aligned
        ((32, 17), -1),
        ((128, 100), -1),
        # rank-2 h_non_aligned
        ((17, 64), -2),
        # rank-3 w_non_aligned
        ((1, 32, 50), -1),
        ((4, 128, 47), -1),
        # rank-3 h_non_aligned
        ((1, 17, 128), -2),
    ],
)
def test_rank2_3_non_aligned(device, shape, dim):
    """Non-tile-aligned shapes with rank-2/3 (masking from Refinement 3)."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)
    expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.operations.softmax.softmax(ttnn_input, dim=dim)
    result = ttnn.to_torch(ttnn_output)

    assert tuple(result.shape) == shape, f"Expected {shape}, got {tuple(result.shape)}"
    assert_with_pcc(result, expected, pcc=0.999)


# ---------------------------------------------------------------------------
# Cross-rank equivalence: rank-2 ≡ rank-4 with leading 1s
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape2d, dim",
    [
        ((32, 64), -1),
        ((128, 512), -2),
    ],
)
def test_rank2_equivalent_to_rank4(device, shape2d, dim):
    """Rank-2 softmax must produce the same result as rank-4 with leading 1s."""
    torch.manual_seed(0)
    torch_input_2d = torch.randn(shape2d, dtype=torch.float32)
    shape4d = (1, 1) + shape2d
    torch_input_4d = torch_input_2d.reshape(shape4d)

    ttnn_input_2d = ttnn.from_torch(torch_input_2d, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input_4d = ttnn.from_torch(torch_input_4d, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out_2d = ttnn.to_torch(ttnn.operations.softmax.softmax(ttnn_input_2d, dim=dim))
    out_4d = ttnn.to_torch(ttnn.operations.softmax.softmax(ttnn_input_4d, dim=dim))

    # Both should match torch.softmax
    expected = torch.softmax(torch_input_2d, dim=dim)
    assert_with_pcc(out_2d, expected, pcc=0.999)
    assert_with_pcc(out_4d.reshape(shape2d), expected, pcc=0.999)
    # Cross-rank equivalence
    assert torch.allclose(out_2d, out_4d.reshape(shape2d), atol=1e-5)


# ---------------------------------------------------------------------------
# Output layout matches input layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 64), (4, 128, 512)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_output_layout_matches_input(device, shape, layout):
    """Output layout must match input layout for rank-2/3 tensors."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=layout, device=device)
    ttnn_output = ttnn.operations.softmax.softmax(ttnn_input, dim=-1)

    assert ttnn_output.layout == ttnn_input.layout


# ---------------------------------------------------------------------------
# Default dim (-1) works without explicit dim argument
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 64), (4, 128, 512)])
def test_default_dim_rank2_3(device, shape):
    """Default dim=-1 (no explicit dim argument) works for rank-2/3."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)
    expected = torch.softmax(torch_input, dim=-1)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.operations.softmax.softmax(ttnn_input)
    result = ttnn.to_torch(ttnn_output)

    assert tuple(result.shape) == shape
    assert_with_pcc(result, expected, pcc=0.999)


# ---------------------------------------------------------------------------
# Negative values (numerical stability)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 64), (4, 128, 512)])
@pytest.mark.parametrize("dim", [-1, -2])
def test_negative_values_rank2_3(device, shape, dim):
    """Negative values must not cause numerical instability for rank-2/3."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32) * 10 - 5  # range [-5, 5]
    expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.operations.softmax.softmax(ttnn_input, dim=dim)
    result = ttnn.to_torch(ttnn_output)

    assert_with_pcc(result, expected, pcc=0.999)


# ---------------------------------------------------------------------------
# Multi-core: larger batch triggers multi-core distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (128, 512),  # rank-2: multi-tile, multiple slabs
        (2, 128, 512),  # rank-3: multi-batch → NC=2 slabs
        (8, 32, 128),  # rank-3: NC=8 slabs → multi-core
    ],
)
@pytest.mark.parametrize("dim", [-1, -2])
def test_multicore_rank2_3(device, shape, dim):
    """Shapes that trigger multi-core distribution for rank-2/3.

    Note: large W (e.g. 4096+) OOMs due to L1 CB budget — that is a
    pre-existing issue addressed by Refinement 5, not a rank-specific bug.
    Here we pick shapes with NC > 1 and reasonable W to stay within L1.
    """
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)
    expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.operations.softmax.softmax(ttnn_input, dim=dim)
    result = ttnn.to_torch(ttnn_output)

    assert tuple(result.shape) == shape
    assert_with_pcc(result, expected, pcc=0.999)
