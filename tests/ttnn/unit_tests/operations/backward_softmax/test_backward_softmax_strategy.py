# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement-2 tests for backward_softmax — input-buffering strategy selection
and per-strategy correctness.

Phase 0 streamed each input tile twice from DRAM (once per pass). Refinement
2 adds a deterministic strategy picker that caches the whole row in L1
across both passes when budget allows:

  1. WHOLE_ROW_DB     — input CBs sized 2 * reduce_dim_tiles. Reader can
                        prefetch lane N+1 while compute is on lane N.
  2. WHOLE_ROW_SB     — input CBs sized reduce_dim_tiles. DRAM reads halved;
                        no cross-lane overlap.
  3. PER_TILE_STREAM  — Phase-0 fallback. Each tile read twice.

These tests verify:
  - The picker chooses the deepest fitting strategy for representative shapes
    (small / medium / large reduce dim).
  - Correctness is preserved across all three strategies — including the
    fallback PER_TILE_STREAM, which exercises the Phase-0 block-loop on a
    reduce dimension that does NOT fit in L1 under whole-row caching.

Tolerance discipline: PCC + relative-RMS rather than torch.allclose with
tight atol. Refinement 2 is about DRAM bandwidth, not precision —
hardware-precision-floor is the subject of Refinement 4. Per-strategy
correctness is checked at PCC >= 0.999 (Phase-0 baseline) so that strategy
regressions show up as PCC drops, distinct from the pre-existing precision
gap.
"""

import pytest
import torch
import ttnn

from ttnn.operations.backward_softmax import backward_softmax
from ttnn.operations.backward_softmax.backward_softmax_program_descriptor import pick_strategy_name
from tests.ttnn.utils_for_testing import check_with_pcc


# -----------------------------------------------------------------------------
# Reference + helpers
# -----------------------------------------------------------------------------


def _torch_reference(grad_output: torch.Tensor, output: torch.Tensor, dim: int) -> torch.Tensor:
    """grad_input = output * (grad_output - sum(output * grad_output, dim))."""
    s = (output * grad_output).sum(dim=dim, keepdim=True)
    return output * (grad_output - s)


def _to_device(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run_and_compare(device, shape, dim, *, pcc=0.999, rel_rms_threshold=0.01):
    torch.manual_seed(0x5EED)
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)
    expected = _torch_reference(torch_dy, torch_y, dim=dim)

    ttnn_dy = _to_device(torch_dy, device)
    ttnn_y = _to_device(torch_y, device)

    ttnn_grad_input = backward_softmax(ttnn_dy, ttnn_y, dim=dim)
    actual = ttnn.to_torch(ttnn_grad_input).float()

    pcc_ok, pcc_msg = check_with_pcc(expected, actual, pcc=pcc)
    rel_rms = (actual - expected).pow(2).mean().sqrt().item() / max(expected.pow(2).mean().sqrt().item(), 1e-12)

    assert pcc_ok, f"PCC failed for shape={shape} dim={dim}: {pcc_msg}"
    assert rel_rms <= rel_rms_threshold, f"rel_rms={rel_rms:.4f} > {rel_rms_threshold} for shape={shape} dim={dim}"


# -----------------------------------------------------------------------------
# Strategy selection — verify the picker is deterministic and budget-driven.
# -----------------------------------------------------------------------------
#
# Boundary cases (Wormhole B0, L1_CB_BUDGET_BYTES = 700 KB, fp32 inputs):
#
#   strategy 1 (WHOLE_ROW_DB) ≤ Wt = 28          (~707 KB total)
#   strategy 2 (WHOLE_ROW_SB)   29 ≤ Wt ≤ 42      (~707 KB total at Wt=42)
#   strategy 3 (PER_TILE_STREAM) Wt ≥ 43          (small CBs, ~200 KB max)


@pytest.mark.parametrize(
    "shape,dim,expected_strategy",
    [
        # ---- dim=-1 (Wt is the reduce dimension) ----
        # Wt=1 — strategy 1 (trivial, fits in L1 trivially).
        pytest.param((1, 1, 32, 32), -1, "WHOLE_ROW_DB", id="Wt=1_DB"),
        # Wt=8 — strategy 1, the most common test shape.
        pytest.param((1, 1, 32, 256), -1, "WHOLE_ROW_DB", id="Wt=8_DB"),
        # Wt=28 — strategy 1 boundary.
        pytest.param((1, 1, 32, 896), -1, "WHOLE_ROW_DB", id="Wt=28_DB_boundary"),
        # Wt=32 — strategy 2 (medium).
        pytest.param((1, 1, 32, 1024), -1, "WHOLE_ROW_SB", id="Wt=32_SB"),
        # Wt=42 — strategy 2 boundary.
        pytest.param((1, 1, 32, 1344), -1, "WHOLE_ROW_SB", id="Wt=42_SB_boundary"),
        # Wt=43 — strategy 3 (per-tile fallback).
        pytest.param((1, 1, 32, 1376), -1, "PER_TILE_STREAM", id="Wt=43_PerTile"),
        # Wt=64 — well into per-tile fallback.
        pytest.param((1, 1, 32, 2048), -1, "PER_TILE_STREAM", id="Wt=64_PerTile"),
        # ---- dim=-2 (Ht is the reduce dimension) — mirror boundaries ----
        pytest.param((1, 1, 1024, 32), -2, "WHOLE_ROW_SB", id="Ht=32_SB_dim_-2"),
        pytest.param((1, 1, 2048, 32), -2, "PER_TILE_STREAM", id="Ht=64_PerTile_dim_-2"),
    ],
)
def test_backward_softmax_strategy_selection(device, shape, dim, expected_strategy):
    """
    The picker is purely a function of shape + dtype + L1 budget. Per the
    contract in op_requirements.md Refinement 2, the selection must be
    deterministic given shape, dtype, and grid — this test pins those
    boundaries so a budget tweak that silently shifts a shape into a
    different strategy fails CI.
    """
    t = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    actual_strategy = pick_strategy_name(t, dim=dim)
    assert (
        actual_strategy == expected_strategy
    ), f"shape={shape} dim={dim}: picker chose {actual_strategy}, expected {expected_strategy}"


# -----------------------------------------------------------------------------
# Per-strategy correctness — drive a representative shape through each
# strategy and verify it matches the torch reference.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,dim,expected_strategy",
    [
        # Strategy 1 (DB) — the bread-and-butter path.
        pytest.param((1, 1, 32, 256), -1, "WHOLE_ROW_DB", id="strategy_1_DB"),
        # Strategy 2 (SB) — fits single-buffered but NOT double-buffered.
        # This is the strategy 2 shape required by op_requirements.md.
        pytest.param((1, 1, 32, 1024), -1, "WHOLE_ROW_SB", id="strategy_2_SB"),
        # Strategy 3 (per-tile fallback) — reduce dim too large to cache.
        pytest.param((1, 1, 32, 2048), -1, "PER_TILE_STREAM", id="strategy_3_PerTile"),
        # Same three strategies along dim=-2 — ensures the column-orientation
        # block shapes are exercised under each buffering regime.
        pytest.param((1, 1, 256, 32), -2, "WHOLE_ROW_DB", id="strategy_1_DB_dim_-2"),
        pytest.param((1, 1, 1024, 32), -2, "WHOLE_ROW_SB", id="strategy_2_SB_dim_-2"),
        pytest.param((1, 1, 2048, 32), -2, "PER_TILE_STREAM", id="strategy_3_PerTile_dim_-2"),
    ],
)
def test_backward_softmax_strategy_correctness(device, shape, dim, expected_strategy):
    """
    Sanity-check that backward_softmax produces a numerically correct result
    under each of the three buffering strategies. The strategy is locked in
    by shape; we assert on the strategy first (so a test failure reads
    "strategy X went numerically wrong" rather than "test mysteriously
    fails after a budget tweak").
    """
    t = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    actual_strategy = pick_strategy_name(t, dim=dim)
    assert actual_strategy == expected_strategy, (
        f"Test guard: shape={shape} dim={dim} picker chose {actual_strategy}, "
        f"expected {expected_strategy} — adjust shape or expectation."
    )

    _run_and_compare(device, shape, dim)


# -----------------------------------------------------------------------------
# Cross-strategy equivalence — same shape across strategies should produce
# numerically equivalent results (modulo the precision-floor noise that's
# pre-existing in pass 1's matmul-based reduce).
# -----------------------------------------------------------------------------


def test_backward_softmax_strategy_2_matches_strategy_1_pcc(device):
    """
    Strategy 1 (DB) and Strategy 2 (SB) share the same compute kernel path —
    only CB sizing differs. Their outputs should be PCC-equivalent for any
    shape that fits both. Forcing the choice from outside is not part of the
    public API (the picker is deterministic), so we can't directly compare
    the two on the same shape. Instead, we sanity-check that both strategies
    deliver Phase-0-quality output on their representative shapes.
    """
    # Strategy 1 representative.
    _run_and_compare(device, (1, 1, 32, 256), -1, pcc=0.9999)
    # Strategy 2 representative.
    _run_and_compare(device, (1, 1, 32, 1024), -1, pcc=0.999)


def test_backward_softmax_multicore_with_per_tile_strategy(device):
    """
    Verify that the PER_TILE_STREAM (strategy-3 fallback) path interacts
    correctly with multi-core distribution: a shape large enough to land on
    strategy 3 AND with multiple lanes spread across the grid.

    Shape (1, 2, 32, 2048) — dim=-1, lanes = N*C*Ht = 2 (small lane count,
    distributed across 2 cores out of the grid). PER_TILE_STREAM strategy.
    """
    shape = (1, 2, 32, 2048)
    dim = -1
    t = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    assert pick_strategy_name(t, dim=dim) == "PER_TILE_STREAM"
    _run_and_compare(device, shape, dim)
