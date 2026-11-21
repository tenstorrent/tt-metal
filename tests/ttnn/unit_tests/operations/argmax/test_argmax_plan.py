# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The tests the design's thread plan requires (op_design.md, Work Distribution, "Tests the plan needs").

(1) every scanner thread of the plan ran (a correct result alone does not show it);
(2) the op after another program, (3) before one of tt-metal's own, each in one device session;
(4) both regimes, pinned, on sub-grids: the result does not depend on `sub_core_grids` (R5).
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.argmax.argmax import _argmax, argmax
from ttnn.operations.argmax.argmax_program_artifacts import (
    ROWS,
    SCAN_THREADS,
    SEGMENTED,
    _grid,
    select_regime,
    split_units,
)

ROWS_SHAPE = (4, 8, 32, 256)
SEG_SHAPE = (1, 1, 2, 151936)


def _input(shape, device, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return x, t


def _check(x, out):
    expected = torch.argmax(x.double(), dim=-1)
    got = ttnn.to_torch(out).to(torch.int64).reshape(expected.shape)
    assert torch.equal(got, expected), f"{(got != expected).sum().item()} indices differ"


def _scanner_of_units(device, shape, sub_core_grids=None):
    """{unit: global scanner id g}, the regime and the unit count, from the host split."""
    rows, w = math.prod(shape[:-1]), shape[-1]
    grid = _grid(device, sub_core_grids)
    regime = select_regime(rows, w, 2, grid.num_cores())
    _, cores, _, t_start, t_count = split_units(grid, regime.num_units)
    owner = {}
    for ordinal, core in enumerate(cores):
        for t in range(SCAN_THREADS):
            for u in range(t_start[core][t], t_start[core][t] + t_count[core][t]):
                owner[u] = ordinal * SCAN_THREADS + t
    assert sorted(owner) == list(range(regime.num_units))
    return owner, regime


def test_every_scanner_thread_ran_rows(device):
    owner, regime = _scanner_of_units(device, ROWS_SHAPE)
    assert regime.name == ROWS
    x, t = _input(ROWS_SHAPE, device)
    out = _argmax(t, -1, probe_scanner_ids=True)
    got = ttnn.to_torch(out).to(torch.int64).flatten().tolist()
    expected = [owner[r] for r in range(regime.num_units)]
    assert got == expected
    assert {g % SCAN_THREADS for g in got} == set(range(SCAN_THREADS))


def test_every_scanner_thread_ran_segmented(device):
    owner, regime = _scanner_of_units(device, SEG_SHAPE)
    assert regime.name == SEGMENTED
    x, t = _input(SEG_SHAPE, device)
    out = _argmax(t, -1, probe_scanner_ids=True)
    got = ttnn.to_torch(out).to(torch.int64).flatten().tolist()
    rows = math.prod(SEG_SHAPE[:-1])
    expected = [sum(owner[r * regime.spr + s] + 1 for s in range(regime.spr)) for r in range(rows)]
    assert got == expected
    assert {g % SCAN_THREADS for g in owner.values()} == set(range(SCAN_THREADS))


@pytest.mark.parametrize("shape", [ROWS_SHAPE, SEG_SHAPE], ids=["rows", "segmented"])
def test_after_another_program(device, shape):
    """(2) a tt-metal program first, then argmax, in one device session."""
    x, t = _input(shape, device)
    tiled = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
    out = argmax(t, dim=-1)
    _check(x, out)
    assert torch.equal(ttnn.to_torch(tiled), x)


@pytest.mark.parametrize("shape", [ROWS_SHAPE, SEG_SHAPE], ids=["rows", "segmented"])
def test_before_a_tt_metal_program(device, shape):
    """(3) argmax, then a tt-metal program of its own (tilize + untilize), in one device session."""
    x, t = _input(shape, device)
    out = argmax(t, dim=-1)
    round_trip = ttnn.to_layout(ttnn.to_layout(t, ttnn.TILE_LAYOUT), ttnn.ROW_MAJOR_LAYOUT)
    out_rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
    _check(x, out)
    _check(x, out_rm)
    assert torch.equal(ttnn.to_torch(round_trip), x)


def test_regimes_on_the_device_grid(device):
    """(4) the regime each shape selects on the full grid."""
    grid = _grid(device, None)
    n = grid.num_cores()
    assert select_regime(math.prod(ROWS_SHAPE[:-1]), ROWS_SHAPE[-1], 2, n).name == ROWS
    assert select_regime(2, 151936, 2, n).name == SEGMENTED
    assert select_regime(32, 4096, 2, n).name == SEGMENTED


def _sub_grid(cols, rows):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])


# (shape, sub-grid, regime): S_total = clusters x 6; SEGMENTED when min(S_total // rows, 2*W // 2048) >= 2.
SUB_GRID_CASES = [
    (SEG_SHAPE, (1, 1), SEGMENTED),  # S_total 6: spr = 3
    (SEG_SHAPE, (4, 2), SEGMENTED),  # S_total 48: spr = 24
    ((1, 1, 32, 64), (1, 1), ROWS),
    ((1, 1, 32, 64), (4, 2), ROWS),
    ((1, 1, 32, 4096), (1, 1), ROWS),  # 6 // 32 = 0
    ((1, 1, 32, 4096), (4, 2), ROWS),  # 48 // 32 = 1
    ((1, 1, 8, 4096), (4, 2), SEGMENTED),  # 48 // 8 = 6, 8192 // 2048 = 4
    ((1, 1, 17, 50), (1, 1), ROWS),
    ((1, 1, 17, 50), (4, 2), ROWS),
]


@pytest.mark.parametrize(
    "shape,grid_shape,expected_regime",
    SUB_GRID_CASES,
    ids=[f"{'x'.join(map(str, s))}-{g[0]}x{g[1]}" for s, g, _ in SUB_GRID_CASES],
)
def test_sub_core_grids_same_result(device, shape, grid_shape, expected_regime):
    """(4) both regimes on 1-cluster and 4x2 sub-grids: the result equals the reference and the full grid's (R5)."""
    sub = _sub_grid(*grid_shape)
    _, regime = _scanner_of_units(device, shape, sub)
    assert regime.name == expected_regime
    x, t = _input(shape, device, seed=1)
    out = argmax(t, dim=-1, sub_core_grids=sub)
    _check(x, out)
    full = argmax(t, dim=-1)
    assert torch.equal(ttnn.to_torch(out), ttnn.to_torch(full))


def test_repeated_calls_reuse_the_semaphore(device):
    """The finalizer resets its semaphore: the same SEGMENTED program run twice gives the same result."""
    x, t = _input(SEG_SHAPE, device, seed=2)
    first = argmax(t, dim=-1)
    second = argmax(t, dim=-1)
    _check(x, first)
    _check(x, second)


def test_positive_dim_alias(device):
    """R4: rank-4 dim=3 is dim=-1."""
    x, t = _input((1, 1, 32, 64), device)
    _check(x, argmax(t, dim=3))


# Streaming: a lowered block cap makes each scanner walk several blocks (IN_DEPTH = 2 in flight), and a cap
# below a row piece splits the piece over column blocks with the running (key, index) carried across them.
STREAM_CASES = [
    (ROWS_SHAPE, None, 1024),  # ROWS, whole rows: 2 rows per block, 3 blocks per scanner
    ((4, 8, 32, 47), None, 192),  # ROWS, whole rows at a 96-B pitch: 2 rows per block
    (ROWS_SHAPE, None, 128),  # ROWS, each 256-element row split over 4 column blocks
    ((1, 1, 32, 4096), (4, 2), 1024),  # ROWS on a sub-grid, 4096-element rows over 8 column blocks
    (SEG_SHAPE, None, 1024),  # SEGMENTED, 1583-element segments over 2 column blocks
    (SEG_SHAPE, (1, 1), 4096),  # SEGMENTED on one cluster, 50646-element segments over 13 blocks
    ((1, 1, 17, 50), None, 64),  # ROWS, 50 elements over 2 column blocks (32 + 18)
]


@pytest.mark.parametrize(
    "shape,grid_shape,cap",
    STREAM_CASES,
    ids=[f"{'x'.join(map(str, s))}-{g[0]}x{g[1] if g else 'full'}-cap{c}" if g else f"{'x'.join(map(str, s))}-full-cap{c}"
         for s, g, c in STREAM_CASES],
)
def test_streaming_blocks(device, shape, grid_shape, cap):
    sub = _sub_grid(*grid_shape) if grid_shape else None
    x, t = _input(shape, device, seed=3)
    _check(x, _argmax(t, -1, sub_core_grids=sub, block_bytes_cap=cap))


def test_streaming_ties_split_across_blocks(device):
    """Ties that straddle column blocks and segments: the lowest index still wins (R1)."""
    shape = (1, 1, 4, 8192)
    x = torch.zeros(shape, dtype=torch.bfloat16)
    x[..., 1000] = 3.0
    x[..., 5000] = 3.0
    x[0, 0, 1, :] = -0.0  # all signed zeros: index 0
    x[0, 0, 2, 7000] = 1.0
    x[0, 0, 2, 7999] = 1.0
    x[0, 0, 3, :] = 0.5
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    for cap in (None, 1024, 64):
        _check(x, _argmax(t, -1, block_bytes_cap=cap))
