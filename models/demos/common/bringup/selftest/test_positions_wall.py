# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F263 (owner 2026-11-09, DeepSeek-V4.1 8x4: one timed run a position put 128k below 51.2k): every position is timed
as the clean wall is - warm-up until a run adds no program and the walls settle, then WALL_RUNS timed runs with
program-cache misses refused, pos_ms_<start> their median. BRINGUP_POSITIONS overrides the spec's positions."""

import types

import pytest

from models.demos.common.bringup.testing import positions as P
from models.demos.common.bringup.testing import profile


class _Mesh:
    """Counts program-cache entries: the first ``cold`` runs of a position each add one. Records misses_allowed."""

    def __init__(self, cold: int):
        self.n, self.cold, self.allowed = 0, cold, []

    def num_program_cache_entries(self):
        return self.n

    def set_program_cache_misses_allowed(self, allowed):
        self.allowed.append(allowed)


class _Model:
    """A device model whose sync advances a fake clock: by the position's next warm-up wall, or, while cache misses are
    refused (the timed runs), by its next timed wall."""

    def __init__(self, mesh, clock, walls, timed):
        self.mesh, self.clock, self.walls, self.timed, self.syncs = mesh, clock, walls, timed, {}

    def new_state(self, seq):
        return types.SimpleNamespace(start=seq - 64)

    def embed(self, tokens):
        return object()

    def layer(self, i, h, start, state):
        self.start = start
        return h

    def final_norm(self, h):
        return h

    def free(self, h):
        pass

    def sync(self):
        k = self.syncs.get(self.start, 0)
        self.syncs[self.start] = k + 1
        if k < self.mesh.cold:
            self.mesh.n += 1
        if self.mesh.allowed and self.mesh.allowed[-1] is False:
            ms = self.timed[self.start].pop(0)
        else:
            w = self.walls[self.start]
            ms = w[min(k, len(w) - 1)]
        self.clock[0] += ms / 1e3


@pytest.fixture
def run(monkeypatch):
    def go(walls, timed, cold=2, env=None):
        got = {}
        mesh, clock = _Mesh(cold), [0.0]
        model = _Model(mesh, clock, walls, timed)
        spec = types.SimpleNamespace(
            data={"target": {"seq": 128, "chunk": 64}},
            num_layers=2,
            layers=lambda: [0, 1],
            get=lambda k, d=None: {"target.chunk": 64, "perf.positions": [0, 64]}.get(k, d),
        )
        monkeypatch.setattr(P.batch, "device_model", lambda *a, **kw: model)
        monkeypatch.setattr(P.time, "time", lambda: clock[0])
        monkeypatch.setattr(P.metrics, "record", lambda k, v, **kw: got.__setitem__(k, v))
        monkeypatch.setattr(P.cpu_bridge, "record", lambda *a, **kw: None)
        monkeypatch.setattr(profile, "WALL_RUNS", 5)
        if env is None:
            monkeypatch.delenv(P.POSITIONS_ENV, raising=False)
        else:
            monkeypatch.setenv(P.POSITIONS_ENV, env)
        rows = P.run_positions(spec, mesh)
        return rows, got, mesh, model

    return go


def test_each_position_is_the_median_of_warm_settled_runs_with_misses_refused(run):
    # compile, 2 cache-filling runs (slow), a falling host, then the plateau; the timed runs hold one outlier each
    warm = [900.0, 300.0, 250.0, 180.0, 140.0] + [100.0] * 30
    walls = {0: warm, 64: [w + 20 for w in warm]}
    timed = {0: [100.0, 100.0, 400.0, 100.0, 101.0], 64: [120.0, 500.0, 119.0, 121.0, 120.0]}
    rows, got, mesh, model = run(walls, timed)
    assert [s for s, _ in rows] == [0, 64]
    assert got["pos_ms_0"] == 100.0 and got["pos_ms_64"] == 120.0  # the median, not the first or the mean
    assert got["pos_ms_runs_0"] == [100.0, 100.0, 400.0, 100.0, 101.0] and got["pos_ms_spread_0"] == 300.0
    assert got["pos_settled_0"] == 1 and got["pos_settled_64"] == 1 and got["pos_0_new_programs"] == 0
    assert mesh.allowed == [False, True, False, True]  # every position's timed runs refused cache misses
    assert not timed[0] and not timed[64]  # exactly WALL_RUNS timed runs each
    # the warm-up ran past the cache fill and the falling host: the plateau only
    assert got["pos_warm_runs_0"] >= 2 * profile.WALL_SETTLE_WINDOW and model.syncs[0] == 1 + got["pos_warm_runs_0"] + 5


def test_walls_that_never_settle_are_flagged(run, monkeypatch):
    monkeypatch.setattr(profile, "WALL_MAX_SETTLE", 3)
    falling = [1000.0 - 37 * i for i in range(30)]
    rows, got, _, _ = run({0: falling}, {0: [100.0] * 5}, env="0")
    assert [s for s, _ in rows] == [0] and got["pos_settled_0"] == 0 and got["pos_ms_0"] == 100.0


def test_the_env_overrides_the_spec_positions(run):
    rows, got, _, _ = run({s: [100.0] * 40 for s in (192, 320)}, {s: [100.0] * 5 for s in (192, 320)}, env="320,192")
    assert [s for s, _ in rows] == [320, 192] and "pos_ms_0" not in got and got["pos_ms_192"] == 100.0
