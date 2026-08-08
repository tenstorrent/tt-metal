# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `parallelism.select_parallelism` — the TP x DP selector that turns per-TP kernel
viability (KernelReport.has_blockers over the grid) into a chosen TP x DP split. Closes the gap where
the tool computed viability but never acted on it (DP was hard-fixed to 1)."""
from scripts.tt_hw_planner.parallelism import select_parallelism

_GRID = [1, 2, 4, 8, 32]


class _FakeReport:
    def __init__(self, grid, blocked):
        self.tp_grid = grid
        self._blocked = set(blocked)

    def has_blockers(self, tp=None):
        return tp in self._blocked


def _sel(chips, blocked):
    pc = select_parallelism(chips, _FakeReport(_GRID, blocked))
    return (pc.tp, pc.dp)


def test_all_viable_fills_with_tp():
    assert _sel(4, []) == (4, 1)


def test_largest_mesh_tp_blocked_falls_to_tp2_dp2():
    assert _sel(4, [4]) == (2, 2)


def test_only_tp1_viable_dp_fills():
    assert _sel(4, [4, 2]) == (1, 4)


def test_single_chip():
    assert _sel(1, []) == (1, 1)


def test_eight_chips_tp8_blocked():
    assert _sel(8, [8]) == (4, 2)


def test_eight_chips_tp8_tp4_blocked():
    assert _sel(8, [8, 4]) == (2, 4)


def test_blocked_degree_not_dividing_chips_is_ignored():
    assert _sel(4, [32]) == (4, 1)


def test_product_always_equals_chips():
    for chips in (1, 2, 4, 8):
        for blocked in ([], [4], [8], [8, 4], [4, 2, 8]):
            pc = select_parallelism(chips, _FakeReport(_GRID, blocked))
            assert pc.tp * pc.dp == chips
