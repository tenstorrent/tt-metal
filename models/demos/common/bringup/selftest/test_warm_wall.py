# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F196 (owner 2026-10-27): the clean wall is timed only once a run adds no program-cache entry, with misses refused
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
    assert w["warm"] and len(w["walls"]) == 4 and w["entries"] == [0, 50, 57, 60, 60]


def test_a_run_that_never_settles_is_not_warm(monkeypatch):
    monkeypatch.setattr(profile, "WALL_MAX_WARM", 5)
    m = _Mesh([1] * 10)
    w = profile.warm_up(m, m.run)
    assert not w["warm"] and len(w["walls"]) == 5


def test_a_mesh_that_cannot_count_falls_back_to_the_fixed_warm_up(monkeypatch):
    monkeypatch.setattr(profile, "WALL_WARM", 3)
    w = profile.warm_up(None, lambda: 1.0)
    assert w["warm"] and len(w["walls"]) == 3


@pytest.mark.parametrize("allowed", [False, True])
def test_misses_allowed_reaches_the_mesh(allowed):
    m = _Mesh([])
    profile.misses_allowed(m, allowed)
    assert m.allowed[-1] is allowed
