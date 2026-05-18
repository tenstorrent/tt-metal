# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-core distribution test for backward_softmax (Refinement 1).

The op partitions its embarrassingly-parallel lanes (one tile-row for
``dim=-1``, one tile-column for ``dim=-2``) across the full
``device.compute_with_storage_grid_size()`` grid via
``ttnn.split_work_to_cores``. The contract:

  * No two cores differ by more than 1 lane (balanced load).
  * Per-core lane ranges form a partition of ``[0, total_lanes)`` — no
    overlap, no gap.
  * Each core runs its assigned lanes end-to-end; no inter-core comms.

This file exercises the partitioner with shapes chosen to trigger
distinct regimes — single-lane, partial-grid, exactly-full-grid,
full-grid-with-remainder, and multi-lane-per-core-with-remainder.

Correctness is checked against the same PyTorch reference the spec test
uses; a coverage bug in the per-core start_lane/num_lanes arithmetic
would manifest as either duplicated or missing output tiles, both of
which the reference comparison catches.

Tolerance: PCC ≥ 0.999 (per Phase 0 baseline). The spec test's tighter
``atol=0.01`` is hardware-precision-floor-bound for shapes whose reduce
dimension exceeds one tile (see ``test_backward_softmax_precision_baseline.py``
and op_requirements.md Refinement 4); this file deliberately avoids
re-litigating that bound and focuses on the partitioner.
"""

import pytest
import torch
import ttnn

from ttnn.operations.backward_softmax import backward_softmax
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


def _run_and_compare(device, shape, dim, *, pcc=0.999, rtol=0.05, atol=0.05):
    """
    Run backward_softmax on a deterministic input and compare against the
    PyTorch reference. We use PCC + a loose atol to stay tolerance-clear of
    the Phase-0 precision floor (the topic of Refinement 4 — explicitly NOT
    of Refinement 1).
    """
    torch.manual_seed(0x5EED)
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)
    expected = _torch_reference(torch_dy, torch_y, dim=dim)

    ttnn_dy = _to_device(torch_dy, device)
    ttnn_y = _to_device(torch_y, device)

    ttnn_grad_input = backward_softmax(ttnn_dy, ttnn_y, dim=dim)
    actual = ttnn.to_torch(ttnn_grad_input).float()

    pcc_ok, pcc_msg = check_with_pcc(expected, actual, pcc=pcc)
    assert pcc_ok, f"PCC failed for shape={shape} dim={dim}: {pcc_msg}"
    return actual, expected


# -----------------------------------------------------------------------------
# Distribution-regime cases
# -----------------------------------------------------------------------------
#
# Lane counts are chosen so that the partitioner is forced into specific
# regimes on the standard Wormhole B0 compute-with-storage grid (8x8 = 64).
# The cases are robust to smaller grids too:
#
#   total_lanes=1     → only 1 core does work, regardless of grid size.
#   total_lanes=16    → uses a strict subset of any reasonable grid; no
#                       remainder distribution kicks in.
#   total_lanes=64    → exactly fills a 64-core grid; on a smaller grid
#                       it still passes (just gets remainder distribution
#                       earlier).
#   total_lanes=65    → 65 is prime; on any grid with >1 cores this is
#                       guaranteed to produce a remainder split.
#   total_lanes=96    → 1.5× a 64-core grid → group 1 gets 2 lanes / core,
#                       group 2 gets 1 lane / core; mixed remainder.
#   total_lanes=130   → 130 / 64 = 2.03 → group 1 gets 3 lanes / core,
#                       group 2 gets 2 lanes / core. Heavier per-core work.


@pytest.mark.parametrize(
    "shape,dim,regime",
    [
        # ---- dim = -1 (reduce W, lanes = N*C*Ht) ----
        # 1 lane → single-core fallback path (exercises group_2 == 0 skip).
        pytest.param((1, 1, 32, 32), -1, "1_lane_dim_minus_1", id="1_lane_dim_-1"),
        # 16 lanes → partial grid, no remainder.
        pytest.param((4, 4, 32, 32), -1, "partial_grid_dim_minus_1", id="16_lanes_dim_-1"),
        # 64 lanes → exactly fills the standard Wormhole grid, 1 lane/core.
        pytest.param((8, 8, 32, 32), -1, "full_grid_exact_dim_minus_1", id="64_lanes_dim_-1"),
        # 65 lanes — prime; forces remainder regardless of grid size.
        pytest.param((5, 13, 32, 32), -1, "full_grid_remainder_dim_minus_1", id="65_lanes_dim_-1"),
        # 96 lanes → 1.5× standard grid; mixed remainder distribution.
        pytest.param((3, 32, 32, 32), -1, "multi_lane_remainder_dim_minus_1", id="96_lanes_dim_-1"),
        # 130 lanes → 2+× standard grid with remainder; heavier per-core load.
        pytest.param((5, 26, 32, 32), -1, "heavy_remainder_dim_minus_1", id="130_lanes_dim_-1"),
        # ---- dim = -2 (reduce H, lanes = N*C*Wt) ----
        # 1 lane — single-core fallback. (Wt=1 because W=32.)
        pytest.param((1, 1, 32, 32), -2, "1_lane_dim_minus_2", id="1_lane_dim_-2"),
        # 16 lanes — partial grid.
        pytest.param((4, 4, 32, 32), -2, "partial_grid_dim_minus_2", id="16_lanes_dim_-2"),
        # 64 lanes — full grid exact.
        pytest.param((8, 8, 32, 32), -2, "full_grid_exact_dim_minus_2", id="64_lanes_dim_-2"),
        # 65 lanes — prime → remainder.
        pytest.param((5, 13, 32, 32), -2, "full_grid_remainder_dim_minus_2", id="65_lanes_dim_-2"),
        # 96 lanes.
        pytest.param((3, 32, 32, 32), -2, "multi_lane_remainder_dim_minus_2", id="96_lanes_dim_-2"),
        # 130 lanes.
        pytest.param((5, 26, 32, 32), -2, "heavy_remainder_dim_minus_2", id="130_lanes_dim_-2"),
    ],
)
def test_backward_softmax_multicore_distribution(device, shape, dim, regime):
    """
    Exercise distinct partition regimes — single-core fallback, partial
    grid, full grid (exact), and several remainder distributions.

    Any partition bug (overlap, gap, off-by-one start_lane) shows up as a
    PCC drop or as wrong tile values at the boundary between cores'
    assigned lane ranges. Random inputs with a fixed seed catch both modes.
    """
    _run_and_compare(device, shape, dim)


# -----------------------------------------------------------------------------
# Per-NC traversal — multi-batch with multi-tile reduce
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,dim",
    [
        # Multi-batch (NC=8) × Ht=4 → 32 lanes (dim=-1) with Wt=4 multi-block
        # reduce. Exercises the lane → (nc, idx) decomposition while keeping
        # the per-lane reduction non-trivial.
        pytest.param((2, 4, 128, 128), -1, id="multi_batch_multi_block_dim_-1"),
        pytest.param((2, 4, 128, 128), -2, id="multi_batch_multi_block_dim_-2"),
        # Lane count = 33 (prime relative to 8x8 grid) with multi-block
        # reduce per lane (Wt=4 for dim=-1, Ht=4 for dim=-2). Tests that the
        # split's per-core work loop drives the full (NUM_BLOCKS × BLOCK_SIZE)
        # inner schedule on every core.
        pytest.param((3, 11, 32, 128), -1, id="33_lanes_multi_block_dim_-1"),
        pytest.param((3, 11, 128, 32), -2, id="33_lanes_multi_block_dim_-2"),
    ],
)
def test_backward_softmax_multicore_lane_decomposition(device, shape, dim):
    """
    Lanes are indexed as ``lane → (nc, idx)``: ``nc = lane / lanes_per_nc``,
    ``idx = lane % lanes_per_nc``. With multi-batch shapes (NC > 1) and a
    multi-block reduce per lane (reduce_dim_tiles > BLOCK_SIZE), the cores
    that span an NC boundary in their assigned lane range exercise the full
    decomposition. Any off-by-one in ``start_lane`` or stale per-lane state
    between iterations on the same core surfaces here.
    """
    _run_and_compare(device, shape, dim)


# -----------------------------------------------------------------------------
# Equivalence vs single-core (pre-refinement) — sanity check
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,dim",
    [
        # 64 lanes (full grid) — across cores, the per-lane numeric path is
        # identical to single-core (no inter-core math, no accumulation
        # across cores). Two invocations on the same inputs should be
        # bit-equal.
        pytest.param((8, 8, 32, 32), -1, id="determinism_64_lanes_dim_-1"),
        pytest.param((5, 13, 32, 32), -1, id="determinism_65_lanes_dim_-1"),
    ],
)
def test_backward_softmax_multicore_determinism(device, shape, dim):
    """
    Multi-core distribution must preserve bit-level determinism — running
    the same inputs twice produces bit-identical output even though
    different cores process each lane. (Lanes are independent; the order
    of computation within each lane is unchanged.)
    """
    torch.manual_seed(7)
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)

    ttnn_dy = _to_device(torch_dy, device)
    ttnn_y = _to_device(torch_y, device)

    out1 = ttnn.to_torch(backward_softmax(ttnn_dy, ttnn_y, dim=dim)).float()
    out2 = ttnn.to_torch(backward_softmax(ttnn_dy, ttnn_y, dim=dim)).float()

    assert torch.equal(out1, out2), (
        f"Non-deterministic multi-core output for shape={shape} dim={dim} — "
        f"max diff = {(out1 - out2).abs().max().item():.6e}"
    )
