# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for toy_binary_in_place — validates the add_in_place and regular add
compute helpers across broadcast modes, data formats, and shapes.

Assertions:
  1. Shape preserved: output.shape == input_a.shape
  2. In-place ↔ non-in-place equivalence: both produce identical results (tight PCC)
  3. Correctness vs PyTorch reference: format-specific PCC threshold
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.toy_binary_in_place import toy_binary_in_place


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# PCC thresholds per data format — mirrors the tolerances used in binary_ng tests
FORMAT_PCC = {
    ttnn.bfloat16: 0.9999,
    ttnn.bfloat8_b: 0.999,
}

# Equivalence threshold between in-place and non-in-place paths.
# Both read the same data and do the same math; any difference is due to
# the extra copy phase in the in-place path and potential rounding from
# the additional pack/unpack cycle.
EQUIVALENCE_PCC = {
    ttnn.bfloat16: 0.9999,
    ttnn.bfloat8_b: 0.998,
}


def b_shape_for_broadcast(a_shape, broadcast_mode):
    """Return the expected B tensor shape for a given broadcast mode."""
    H, W = a_shape[-2], a_shape[-1]
    prefix = a_shape[:-2]
    if broadcast_mode == "none":
        return (*prefix, H, W)
    elif broadcast_mode == "row":
        return (*prefix, 32, W)
    elif broadcast_mode == "col":
        return (*prefix, H, 32)
    else:  # scalar
        return (*prefix, 32, 32)


def pytorch_reference(torch_a, torch_b, broadcast_mode):
    """Compute the PyTorch reference for A + B with hardware broadcast semantics.

    Hardware broadcast reads specific tile regions from B:
      ROW:    B[..., :1, :] broadcast across H
      COL:    B[..., :, :1] broadcast across W
      SCALAR: B[..., :1, :1] broadcast everywhere
    """
    a_f = torch_a.float()
    if broadcast_mode == "none":
        return a_f + torch_b.float()
    elif broadcast_mode == "row":
        return a_f + torch_b[..., :1, :].float()
    elif broadcast_mode == "col":
        return a_f + torch_b[..., :, :1].float()
    else:  # scalar
        return a_f + torch_b[..., :1, :1].float()


def run_on_device(device, torch_a, torch_b, broadcast_mode, dtype, in_place):
    """Send tensors to device, run the op, return torch output."""
    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_out = toy_binary_in_place(
        ttnn_a,
        ttnn_b,
        broadcast_mode=broadcast_mode,
        in_place=in_place,
    )
    return ttnn.to_torch(ttnn_out)


# ---------------------------------------------------------------------------
# Shape × Broadcast parametrization
# ---------------------------------------------------------------------------

SHAPES_AND_BROADCASTS = [
    # (a_shape, broadcast_mode, test_id)
    # --- NONE: element-wise ---
    ((1, 1, 32, 32), "none", "none_1t"),
    ((1, 1, 64, 64), "none", "none_2x2t"),
    ((1, 1, 32, 96), "none", "none_1x3t"),
    ((1, 1, 96, 32), "none", "none_3x1t"),
    ((1, 1, 128, 128), "none", "none_4x4t"),
    ((1, 1, 32, 128), "none", "none_1x4t"),
    ((1, 1, 128, 32), "none", "none_4x1t"),
    ((1, 1, 64, 128), "none", "none_2x4t"),
    # --- ROW: B is [1, Wt] tiles, broadcast across H ---
    ((1, 1, 32, 32), "row", "row_1t"),
    ((1, 1, 64, 64), "row", "row_2x2t"),
    ((1, 1, 96, 64), "row", "row_3x2t"),
    ((1, 1, 128, 128), "row", "row_4x4t"),
    ((1, 1, 128, 32), "row", "row_4x1t"),
    # --- COL: B is [Ht, 1] tiles, broadcast across W ---
    ((1, 1, 32, 32), "col", "col_1t"),
    ((1, 1, 64, 64), "col", "col_2x2t"),
    ((1, 1, 64, 96), "col", "col_2x3t"),
    ((1, 1, 128, 128), "col", "col_4x4t"),
    ((1, 1, 32, 128), "col", "col_1x4t"),
    # --- SCALAR: B is [1, 1] tile, broadcast everywhere ---
    ((1, 1, 32, 32), "scalar", "scalar_1t"),
    ((1, 1, 64, 64), "scalar", "scalar_2x2t"),
    ((1, 1, 96, 96), "scalar", "scalar_3x3t"),
    ((1, 1, 128, 128), "scalar", "scalar_4x4t"),
    ((1, 1, 32, 128), "scalar", "scalar_1x4t"),
]


# ---------------------------------------------------------------------------
# Test 1: In-place correctness vs PyTorch reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a_shape, broadcast_mode",
    [(s, b) for s, b, _ in SHAPES_AND_BROADCASTS],
    ids=[tid for _, _, tid in SHAPES_AND_BROADCASTS],
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["bf16", "bf8b"],
)
def test_in_place_correctness(device, a_shape, broadcast_mode, dtype):
    """In-place add matches the PyTorch reference within format-specific PCC."""
    torch.manual_seed(42)
    torch_a = torch.randn(a_shape, dtype=torch.bfloat16)
    torch_b = torch.randn(b_shape_for_broadcast(a_shape, broadcast_mode), dtype=torch.bfloat16)

    expected = pytorch_reference(torch_a, torch_b, broadcast_mode)
    actual = run_on_device(device, torch_a, torch_b, broadcast_mode, dtype, in_place=True)

    # Assert 1: shape preserved
    assert list(actual.shape) == list(
        expected.shape
    ), f"Shape mismatch: actual={list(actual.shape)}, expected={list(expected.shape)}"

    # Assert 2: numerical correctness
    assert_with_pcc(expected, actual.float(), FORMAT_PCC[dtype])


# ---------------------------------------------------------------------------
# Test 2: Non-in-place correctness vs PyTorch reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a_shape, broadcast_mode",
    [(s, b) for s, b, _ in SHAPES_AND_BROADCASTS],
    ids=[tid for _, _, tid in SHAPES_AND_BROADCASTS],
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["bf16", "bf8b"],
)
def test_non_in_place_correctness(device, a_shape, broadcast_mode, dtype):
    """Normal (non-in-place) add matches the PyTorch reference."""
    torch.manual_seed(42)
    torch_a = torch.randn(a_shape, dtype=torch.bfloat16)
    torch_b = torch.randn(b_shape_for_broadcast(a_shape, broadcast_mode), dtype=torch.bfloat16)

    expected = pytorch_reference(torch_a, torch_b, broadcast_mode)
    actual = run_on_device(device, torch_a, torch_b, broadcast_mode, dtype, in_place=False)

    # Assert 1: shape preserved
    assert list(actual.shape) == list(
        expected.shape
    ), f"Shape mismatch: actual={list(actual.shape)}, expected={list(expected.shape)}"

    # Assert 2: numerical correctness
    assert_with_pcc(expected, actual.float(), FORMAT_PCC[dtype])


# ---------------------------------------------------------------------------
# Test 3: In-place produces the same result as non-in-place
#
# This is the KEY in-place assertion: the in-place path (copy→modify→copy)
# must yield the same values as the direct path (add to separate CB).
# Any difference means the pop/push cycle corrupted data.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a_shape, broadcast_mode",
    [(s, b) for s, b, _ in SHAPES_AND_BROADCASTS],
    ids=[tid for _, _, tid in SHAPES_AND_BROADCASTS],
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["bf16", "bf8b"],
)
def test_in_place_matches_non_in_place(device, a_shape, broadcast_mode, dtype):
    """In-place and non-in-place paths produce equivalent results.

    This is the definitive in-place correctness check: if the in-place
    pop/push cycle on the CB introduces any data corruption, tile reordering,
    or missed tiles, the in-place result will diverge from the non-in-place
    result even though both compute the same A + B.
    """
    torch.manual_seed(42)
    torch_a = torch.randn(a_shape, dtype=torch.bfloat16)
    torch_b = torch.randn(b_shape_for_broadcast(a_shape, broadcast_mode), dtype=torch.bfloat16)

    out_in_place = run_on_device(device, torch_a, torch_b, broadcast_mode, dtype, in_place=True)
    out_normal = run_on_device(device, torch_a, torch_b, broadcast_mode, dtype, in_place=False)

    # Assert 1: shapes identical
    assert list(out_in_place.shape) == list(
        out_normal.shape
    ), f"Shape divergence: in_place={list(out_in_place.shape)}, normal={list(out_normal.shape)}"

    # Assert 2: values equivalent (tight PCC — any divergence is a bug)
    assert_with_pcc(out_normal.float(), out_in_place.float(), EQUIVALENCE_PCC[dtype])


# ---------------------------------------------------------------------------
# Test 4: Large shapes stress test (in-place only, bfloat16)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a_shape, broadcast_mode",
    [
        pytest.param((1, 1, 256, 256), "none", id="large_none_8x8t"),
        pytest.param((1, 1, 256, 256), "row", id="large_row_8x8t"),
        pytest.param((1, 1, 256, 256), "col", id="large_col_8x8t"),
        pytest.param((1, 1, 256, 256), "scalar", id="large_scalar_8x8t"),
        pytest.param((1, 1, 64, 256), "none", id="large_none_2x8t"),
        pytest.param((1, 1, 256, 64), "none", id="large_none_8x2t"),
    ],
)
def test_large_shape_in_place(device, a_shape, broadcast_mode):
    """Stress test with larger tile grids to verify the in-place pop/push
    cycle handles many tiles without reordering or corruption."""
    torch.manual_seed(123)
    torch_a = torch.randn(a_shape, dtype=torch.bfloat16)
    torch_b = torch.randn(b_shape_for_broadcast(a_shape, broadcast_mode), dtype=torch.bfloat16)

    expected = pytorch_reference(torch_a, torch_b, broadcast_mode)
    actual = run_on_device(device, torch_a, torch_b, broadcast_mode, ttnn.bfloat16, in_place=True)

    assert list(actual.shape) == list(expected.shape)
    assert_with_pcc(expected, actual.float(), 0.9999)
