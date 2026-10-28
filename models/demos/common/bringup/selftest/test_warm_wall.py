# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F196 / F200 (owner 2026-10-27): the clean wall is timed only once a run adds no program-cache entry, with misses refused
during the timed runs (DeepSeek-V4.1 galaxy_8x4: the first timed run after compile took 26.3 s, a later one 15.8 s)."""

import pytest

from models.demos.common.bringup.testing import profile


class _Mesh:
    def __init__(self, adds):
        self.n, self.adds, self.allowed = 0, list(adds), [True]

    def num_program_cache_entries(self):
        return self.n

    def set_program_cache_misses_allowed(self, allowed):
        self.allowed.append(allowed)

    def run(self):
        self.n += self.adds.pop(0) if self.adds else 0
        return 1.0


def test_warm_is_the_first_run_that_adds_no_entry():
    m = _Mesh([50, 7, 3, 0, 5])
    w = profile.warm_up(m, m.run)
    assert w["warm"] and w["cache_warm_runs"] == 4 and w["entries"] == [0, 50, 57, 60, 60]
    assert w["settled"] and len(w["walls"]) == profile.WALL_SETTLE_WINDOW  # (equal walls: one settle run)


def test_a_run_that_never_settles_is_not_warm(monkeypatch):
    monkeypatch.setattr(profile, "WALL_MAX_WARM", 5)
    m = _Mesh([1] * 10)
    w = profile.warm_up(m, m.run)
    assert not w["warm"] and len(w["walls"]) == 5


def test_a_mesh_that_cannot_count_falls_back_to_the_fixed_warm_up(monkeypatch):
    monkeypatch.setattr(profile, "WALL_WARM", 3)
    w = profile.warm_up(None, lambda: 1.0)
    assert w["warm"] and w["cache_warm_runs"] == 3 and w["settled"]


@pytest.mark.parametrize("allowed", [False, True])
def test_misses_allowed_reaches_the_mesh(allowed):
    m = _Mesh([])
    profile.misses_allowed(m, allowed)
    assert m.allowed[-1] is allowed


def test_a_timed_run_that_built_programs_is_reported_not_warm(monkeypatch, capsys):
    """F197: every timed profile / position run records the programs it built; one that built any says NOT WARM."""
    got = {}
    monkeypatch.setattr(profile.metrics, "record", lambda k, v, **kw: got.__setitem__(k, v))
    m = _Mesh([])
    m.n = 120
    profile.record_warm(m, 100, "profiled")
    assert got["profiled_new_programs"] == 20 and "NOT WARM" in capsys.readouterr().out
    profile.record_warm(m, 120, "plain")
    assert got["plain_new_programs"] == 0 and "NOT WARM" not in capsys.readouterr().out
    profile.record_warm(None, None, "pos_0")
    assert got["pos_0_new_programs"] == -1


def test_the_walls_must_settle_after_the_cache_is_warm():
    """F200 (owner 2026-10-27): DeepSeek-V4.1 2x2 flat path, cache warm after 1 run, then walls falling 142 -> 118 ms;
    the timed median was 138.0 where the settled host runs ~119."""
    flat = iter([142.6, 142.6, 141.3, 148.3, 138.0, 138.9, 123.6, 118.0, 119.0, 118.9, 123.5, 119.0])
    m = _Mesh([40, 0])
    w = profile.warm_up(m, lambda: (m.run(), next(flat))[1])
    assert w["cache_warm_runs"] == 2 and w["settled"] and w["walls"][-5:] == [123.6, 118.0, 119.0, 118.9, 123.5]


def test_walls_that_never_settle_are_flagged_not_refused(monkeypatch):
    monkeypatch.setattr(profile, "WALL_MAX_SETTLE", 4)
    walls = iter(range(100, 0, -7))
    m = _Mesh([0])
    w = profile.warm_up(m, lambda: (m.run(), next(walls))[1])
    assert w["warm"] and not w["settled"] and len(w["walls"]) == 1 + 4


def test_settled_needs_a_full_window_within_the_percentage():
    assert not profile.settled([100.0] * (profile.WALL_SETTLE_WINDOW - 1))
    assert profile.settled([200.0, 100.0, 102.0, 101.0, 104.0, 100.5])
    assert not profile.settled([100.0, 102.0, 101.0, 104.0, 106.0])
