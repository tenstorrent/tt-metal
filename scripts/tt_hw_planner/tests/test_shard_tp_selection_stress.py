"""Stress-test the mesh->TP auto-selection logic that drives shard-aware bring-up:
  - _mesh_chips: parse '2,4' / '2x4' / '1x1' -> chip count
  - select_parallelism: pick the LARGEST kernel-viable TP that DIVIDES chips; dp = chips//tp
  - _derive_shard_tp semantics: 1 chip (or nothing viable) -> TP=1 (Phase 2 stays off)

Verifies the CONTRACT (validity), and documents the known non-optimality (it maximizes TP;
it does not pick the fastest TP). No device / no HF probe needed — select_parallelism is fed a
synthetic kernel report.
"""

from __future__ import annotations

from scripts.tt_hw_planner._cli_helpers.bringup_cc import _mesh_chips
from scripts.tt_hw_planner.parallelism import select_parallelism


class _KR:
    """Synthetic KernelReport: tp_grid + a set of blocked degrees."""

    def __init__(self, tp_grid, blocked=()):
        self.tp_grid = list(tp_grid)
        self._blocked = set(blocked)

    def has_blockers(self, tp):
        return tp in self._blocked


def test_mesh_chips_parsing():
    assert _mesh_chips("2,4") == 8
    assert _mesh_chips("2x4") == 8
    assert _mesh_chips("2X4") == 8
    assert _mesh_chips("1x1") == 1
    assert _mesh_chips("1,4") == 4
    assert _mesh_chips("4") == 4
    assert _mesh_chips(None) == 1
    assert _mesh_chips("") == 1
    assert _mesh_chips("garbage") == 1


def test_single_chip_is_tp1():
    assert (select_parallelism(1, _KR([1, 2, 4, 8])).tp, select_parallelism(1, _KR([1, 2, 4, 8])).dp) == (1, 1)


def test_picks_largest_viable_divisor():
    pc = select_parallelism(8, _KR([1, 2, 4, 8]))
    assert (pc.tp, pc.dp) == (8, 1)


def test_skips_blocked_top_degree():
    pc = select_parallelism(8, _KR([1, 2, 4, 8], blocked={8}))
    assert (pc.tp, pc.dp) == (4, 2)


def test_falls_to_tp1_when_all_blocked():
    pc = select_parallelism(8, _KR([1, 2, 4, 8], blocked={2, 4, 8}))
    assert (pc.tp, pc.dp) == (1, 8)


def test_non_power_of_two_chips():
    pc = select_parallelism(6, _KR([1, 2, 3, 6]))
    assert (pc.tp, pc.dp) == (6, 1)
    pc2 = select_parallelism(6, _KR([1, 2, 3, 6], blocked={6}))
    assert (pc2.tp, pc2.dp) == (3, 2)


def test_grid_degree_not_dividing_chips_is_skipped():
    pc = select_parallelism(8, _KR([1, 3, 5]))
    assert (pc.tp, pc.dp) == (1, 8)


def test_empty_or_missing_grid_defaults_tp1():
    pc = select_parallelism(8, _KR([]))
    assert (pc.tp, pc.dp) == (1, 8)


def test_stress_contract_holds_over_many_cases():
    n_ok = 0
    for chips in range(1, 65):
        for gi in range(8):
            grid = [d for d in (1, 2, 3, 4, 6, 8, 16, 32) if d <= chips]
            blocked = {g for j, g in enumerate(grid) if (j + gi) % 3 == 0 and g > 1}
            pc = select_parallelism(chips, _KR(grid, blocked))
            assert chips % pc.tp == 0, (chips, grid, blocked, pc.tp)
            assert pc.dp == chips // pc.tp
            if pc.tp != 1:
                assert pc.tp not in blocked
                bigger_viable = [g for g in grid if g > pc.tp and chips % g == 0 and g not in blocked]
                assert not bigger_viable, (chips, grid, blocked, pc.tp, bigger_viable)
            n_ok += 1
    assert n_ok == 64 * 8


def test_documents_non_optimality_max_tp_not_fastest():
    pc = select_parallelism(8, _KR([1, 2, 4, 8]))
    assert pc.tp == 8
