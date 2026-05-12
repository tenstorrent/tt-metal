# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for glu_fused — Gated Linear Unit (last-dim split) as a single
fused TTNN kernel.

Math under test:
    glu_fused(x) == torch.nn.functional.glu(x, dim=-1)
                 == x[..., :W/2] * sigmoid(x[..., W/2:])

The operation is a single fused TTNN kernel that folds the TTNN composite
``slice(...) + slice(...) + sigmoid(...) + multiply(...)`` into one
``ttnn.generic_op`` dispatch — see ``op_design.md`` next to the operation
source. No ``ttnn::slice`` anywhere in the implementation; the split happens
at the tile-id level inside the reader.

This test is the immutable spec for the operation. The implementer must NOT
modify this file. If a parametrized case is impossible under the Phase 0
constraints (float32, TILE_LAYOUT, rank 4, W divisible by 64, H divisible
by 32), the implementer should fix the kernel rather than relax the test.

Coverage:
- Multiple shapes (single output tile, multi-tile W, multi-tile H, non-square,
  multi-batch) including shapes that span more than one core.
- Numerical correctness against ``torch.nn.functional.glu`` (PCC ≥ 0.999,
  max abs ≤ 0.05, rel RMS ≤ 1e-3) on N(0, 1) random inputs.
- Output dtype / shape / layout / memory-config preservation.
- Positional and keyword call patterns.
- Negative tests: invalid dtype, wrong layout, wrong rank, W not divisible
  by 64, H not divisible by 32.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.glu_fused import glu_fused


# Phase-0 tolerances per the task spec.
PCC_THRESHOLD = 0.999
MAX_ABS_THRESHOLD = 0.05
REL_RMS_THRESHOLD = 1e-3


def _torch_reference(x: torch.Tensor) -> torch.Tensor:
    """Reference: torch.nn.functional.glu split along the last dim."""
    return torch.nn.functional.glu(x, dim=-1)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Pearson correlation coefficient over flattened tensors.

    Matches the standard definition used elsewhere in tt-metal acceptance
    tests: ``cov(a, b) / (std(a) * std(b))`` over the flattened tensors,
    in fp64 to avoid the metric itself losing precision near 1.0.
    """
    a64 = a.flatten().double()
    b64 = b.flatten().double()
    a_mean = a64.mean()
    b_mean = b64.mean()
    a_centered = a64 - a_mean
    b_centered = b64 - b_mean
    denom = (a_centered.norm() * b_centered.norm()).item()
    if denom == 0.0:
        # Both perfectly constant — treat as a perfect match if they're equal,
        # otherwise fall through and let the abs/rel checks catch the issue.
        return 1.0 if torch.allclose(a64, b64) else 0.0
    return ((a_centered * b_centered).sum() / denom).item()


def _rel_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative RMS error = sqrt(mean((a-b)^2)) / sqrt(mean(b^2))."""
    diff = actual.double() - expected.double()
    num = diff.pow(2).mean().sqrt().item()
    den = expected.double().pow(2).mean().sqrt().item()
    return num / max(den, 1e-12)


def _assert_glu_close(actual: torch.Tensor, expected: torch.Tensor, shape):
    """Run all three Phase-0 tolerance checks with a unified failure message."""
    pcc = _pcc(actual, expected)
    max_abs = (actual.double() - expected.double()).abs().max().item()
    rel_rms = _rel_rms(actual, expected)

    detail = (
        f"Mismatch for shape={shape}:\n"
        f"  PCC       = {pcc:.6f}  (need >= {PCC_THRESHOLD})\n"
        f"  max abs   = {max_abs:.6f}  (need <= {MAX_ABS_THRESHOLD})\n"
        f"  rel RMS   = {rel_rms:.6e}  (need <= {REL_RMS_THRESHOLD:.1e})\n"
        f"  actual.flat[:6]   = {actual.flatten()[:6].tolist()}\n"
        f"  expected.flat[:6] = {expected.flatten()[:6].tolist()}"
    )
    assert pcc >= PCC_THRESHOLD, detail
    assert max_abs <= MAX_ABS_THRESHOLD, detail
    assert rel_rms <= REL_RMS_THRESHOLD, detail


def _make_input(shape, seed: int = 42) -> torch.Tensor:
    """Deterministic N(0, 1) input — matches the task spec's noise model."""
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Positive cases — bulk numerical correctness vs torch.nn.functional.glu
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        # W = 64  → each half = 32 → 1 output tile per (n, c, h_tile).
        pytest.param((1, 1, 32, 64), id="single_output_tile"),
        # W = 128 → each half = 64 → 2 output tiles per (n, c, h_tile).
        pytest.param((1, 1, 32, 128), id="multi_tile_W"),
        # Tall: many tile-rows, one tile-col per half.
        pytest.param((1, 1, 256, 64), id="multi_tile_H"),
        # Larger W: 8 output tiles per row → exercises mid-row split offsets.
        pytest.param((1, 1, 32, 512), id="wide_W"),
        # Non-square: H != W, both halves > 1 tile.
        pytest.param((1, 1, 64, 256), id="non_square_64x256"),
        pytest.param((1, 1, 256, 128), id="non_square_256x128"),
        # Multi-batch: spreads work across multiple cores (NC * Ht * Wt_half
        # tiles total). On Wormhole's 8x8 grid, this exercises the two-group
        # remainder path in split_work_to_cores.
        pytest.param((2, 4, 64, 128), id="multi_batch"),
        # Larger multi-batch: comfortably exceeds 64 cores worth of work.
        pytest.param((2, 2, 128, 256), id="multi_batch_large"),
    ],
)
def test_glu_fused_correctness(device, shape):
    """
    glu_fused matches torch.nn.functional.glu(x, dim=-1) within the Phase-0
    tolerance band on random N(0, 1) inputs.
    """
    torch_input = _make_input(shape, seed=42)
    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = glu_fused(ttnn_input)

    # Metadata sanity.
    expected_shape = (shape[0], shape[1], shape[2], shape[3] // 2)
    assert tuple(ttnn_output.shape) == expected_shape, (
        f"Output shape {tuple(ttnn_output.shape)} != expected {expected_shape} " f"(input shape {shape})"
    )
    assert ttnn_output.dtype == ttnn.float32, f"Output dtype {ttnn_output.dtype} != float32"
    assert ttnn_output.layout == ttnn.TILE_LAYOUT, f"Output layout {ttnn_output.layout} != TILE_LAYOUT"

    actual = ttnn.to_torch(ttnn_output).float()
    _assert_glu_close(actual, torch_expected, shape)


# -----------------------------------------------------------------------------
# Call-pattern tests — positional and keyword
# -----------------------------------------------------------------------------


def test_glu_fused_positional_call(device):
    """glu_fused(t) — positional argument call style works."""
    shape = (1, 1, 32, 64)
    torch_input = _make_input(shape)
    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = glu_fused(ttnn_input)  # positional
    actual = ttnn.to_torch(ttnn_output).float()
    _assert_glu_close(actual, torch_expected, shape)


def test_glu_fused_keyword_call(device):
    """glu_fused(input_tensor=t) — keyword argument call style works."""
    shape = (1, 1, 32, 64)
    torch_input = _make_input(shape)
    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = glu_fused(input_tensor=ttnn_input)  # keyword
    actual = ttnn.to_torch(ttnn_output).float()
    _assert_glu_close(actual, torch_expected, shape)


# -----------------------------------------------------------------------------
# Deterministic structural check — the SPLIT is correctly placed
# -----------------------------------------------------------------------------


def test_glu_fused_split_offset_arange(device):
    """
    Deterministic input ``arange``-style — every input element has a unique
    value, so any off-by-one in the tile-id arithmetic that drives the split
    shows up as a wrong specific value at a recognizable location.

    Using ``x = arange / scale`` keeps inputs in a numerically benign range
    for the sigmoid + mul pipeline (sigmoid saturates outside ±10 or so).
    """
    shape = (1, 1, 32, 128)  # W=128, halves of 64 each → 2 output tiles per row
    n_elem = math.prod(shape)
    torch_input = (torch.arange(n_elem, dtype=torch.float32) / n_elem).reshape(shape)
    # Re-center so the values exercise both halves of sigmoid (not all >0.5).
    torch_input = torch_input - 0.5

    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = glu_fused(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()
    _assert_glu_close(actual, torch_expected, shape)


# -----------------------------------------------------------------------------
# Negative cases (Python-side validation)
# -----------------------------------------------------------------------------


def _make_tensor(device, shape, *, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    """Helper to build on-device tensors with controlled dtype / layout."""
    torch.manual_seed(42)
    torch_dtype = {
        ttnn.float32: torch.float32,
        ttnn.bfloat16: torch.bfloat16,
    }.get(dtype, torch.float32)

    return ttnn.from_torch(
        torch.randn(shape, dtype=torch_dtype),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_glu_fused_rejects_bf16_dtype(device):
    """Phase 0: float32 only. bfloat16 inputs must be rejected."""
    bf16_input = _make_tensor(device, (1, 1, 32, 64), dtype=ttnn.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        glu_fused(bf16_input)


def test_glu_fused_rejects_row_major_layout(device):
    """TILE_LAYOUT only. ROW_MAJOR_LAYOUT must be rejected."""
    rm_input = _make_tensor(device, (1, 1, 32, 64), layout=ttnn.ROW_MAJOR_LAYOUT)
    with pytest.raises((ValueError, RuntimeError)):
        glu_fused(rm_input)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 64), id="rank_2"),
        pytest.param((1, 32, 64), id="rank_3"),
        pytest.param((1, 1, 1, 32, 64), id="rank_5"),
    ],
)
def test_glu_fused_rejects_wrong_rank(device, shape):
    """
    Phase 0 requires rank == 4 (N, C, H, W). Anything else must be rejected.
    Construction itself may raise (e.g. ttnn.from_torch rejecting rank<2 in
    TILE_LAYOUT) — that is also acceptable; the precondition is enforced
    *somewhere* in the dispatch path.
    """
    torch.manual_seed(42)
    try:
        bad_input = ttnn.from_torch(
            torch.randn(shape, dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    except (ValueError, RuntimeError):
        return
    with pytest.raises((ValueError, RuntimeError)):
        glu_fused(bad_input)


@pytest.mark.parametrize(
    "shape",
    [
        # W = 32 → half = 16, NOT tile-aligned. Must be rejected.
        pytest.param((1, 1, 32, 32), id="W_div_32_but_not_64"),
        # W = 96 → half = 48, NOT tile-aligned.
        pytest.param((1, 1, 32, 96), id="W_96_half_48"),
    ],
)
def test_glu_fused_rejects_W_not_divisible_by_64(device, shape):
    """
    Phase 0 requires W % 64 == 0 (each half must be tile-aligned). W that's
    divisible by 32 but not by 64 — e.g. 32, 96, 160 — is rejected because
    the second half would not start at a tile boundary.
    """
    bad_input = _make_tensor(device, shape)
    with pytest.raises((ValueError, RuntimeError)):
        glu_fused(bad_input)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 30, 64), id="H_not_tile_aligned_30"),
        pytest.param((1, 1, 16, 64), id="H_not_tile_aligned_16"),
    ],
)
def test_glu_fused_rejects_H_not_divisible_by_32(device, shape):
    """
    Phase 0 requires H % 32 == 0. Non-tile-aligned H must be rejected.
    Note: ttnn.from_torch with TILE_LAYOUT internally pads to 32; the
    validator must still flag the underlying logical shape.
    """
    bad_input = _make_tensor(device, shape)
    with pytest.raises((ValueError, RuntimeError)):
        glu_fused(bad_input)
