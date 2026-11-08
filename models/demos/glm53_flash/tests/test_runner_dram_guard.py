# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the prefill runtime's DRAM guard and growth note (tt-blaze G53F-0313): a fake mesh device and
DRAM view, the real ``GLM53FlashPrefillRuntime`` methods (no model, no chips)."""

from __future__ import annotations

import pytest

ttnn = pytest.importorskip("ttnn")
from models.demos.glm53_flash.tt.runners import runtime as RT  # noqa: E402


class _View:
    def __init__(self, held_mib, free_mib):
        self.total_bytes_allocated_per_bank = int(held_mib * 2**20)
        self.largest_contiguous_bytes_free_per_bank = int(free_mib * 2**20)


class _Dram:
    def __init__(self):
        self.held, self.free, self.progs, self.clears, self.syncs = 3786.0, 276.0, 2345, 0, 0


class _Mesh:
    def __init__(self, d):
        self.d = d

    def num_program_cache_entries(self):
        return self.d.progs

    def clear_program_cache(self):
        self.d.clears += 1
        self.d.progs = 0
        self.d.held -= 150
        self.d.free += 150


class _Leaf:  # stands for a model module: the walker follows objects of "models." modules
    pass


_Leaf.__module__ = "models.demos.glm53_flash.tt.indexer"


@pytest.fixture
def rt(monkeypatch):
    d = _Dram()
    monkeypatch.setattr(RT.ttnn, "get_memory_view", lambda dev, bt: _View(d.held, d.free))
    monkeypatch.setattr(RT.ttnn, "synchronize_device", lambda dev: setattr(d, "syncs", d.syncs + 1))
    for k in ("GLM53F_RUNNER_DRAM_GUARD_SELFTEST", "GLM53F_RUNNER_DRAM_GUARD_MIB", "GLM53_MODEL_TRACE_MB"):
        monkeypatch.delenv(k, raising=False)
    a, b = _Leaf(), _Leaf()
    a._const_cache = {("bounds", 1): object(), ("tail", 2): object()}
    b._cu_windows = {5: object()}
    b._prefill_views = {("pools", 0): object()}
    a.child = b
    model = _Leaf()
    model.decoders = {0: a}
    r = object.__new__(RT.GLM53FlashPrefillRuntime)
    r.mesh_device, r.model, r._async = _Mesh(d), model, False
    return r, d, a, b


def test_growth_note_reports_only_changes(rt):
    r, d, a, _ = rt
    note = r._growth_note()
    assert "programs 2345" in note and "device consts 3" in note
    assert r._growth_note() == ""
    a._const_cache[("bounds", 3)] = object()
    assert "device consts 4" in r._growth_note()


def test_guard_clears_below_the_floor_only(rt):
    r, d, a, b = rt
    r._arm_dram_guard()
    assert r._dram_floor_mib == pytest.approx(276 - 64)
    r._dram_guard(0)
    assert d.clears == 0
    d.free, d.held = 100.0, 3950.0
    r._dram_guard(12288)
    assert (d.clears, d.syncs, r._guard_clears) == (1, 1, 1)
    assert not a._const_cache and not b._cu_windows
    assert len(b._prefill_views) == 1  # page-table views are state, not constants: kept


def test_selftest_forces_one_clear_and_keeps_the_floor(rt, monkeypatch):
    r, d, a, _ = rt
    monkeypatch.setenv("GLM53F_RUNNER_DRAM_GUARD_SELFTEST", "1")
    r._arm_dram_guard()
    assert d.clears == 1 and not a._const_cache
    assert r._dram_floor_mib == pytest.approx(212)


@pytest.mark.parametrize("env", [{"GLM53_MODEL_TRACE_MB": "512"}, {"GLM53F_RUNNER_DRAM_GUARD_MIB": "0"}])
def test_guard_refuses_to_arm(rt, monkeypatch, env):
    r, d, _, _ = rt
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    r._arm_dram_guard()
    assert r._dram_floor_mib is None
    d.free = 0.0
    r._dram_guard(0)
    assert d.clears == 0
