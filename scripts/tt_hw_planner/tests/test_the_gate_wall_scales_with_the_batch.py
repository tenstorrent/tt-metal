# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pin: a flat gate wall and an enforced --batch cannot both hold.

The e2e gate's hang wall was added 2026-07-04 ("cap the per-gate pytest timeout at
E2E_GATE_HANG_TIMEOUT (default 2700s / 45min) instead of the full 4h agent budget, so a fabric hang
is caught in minutes not hours"). 2700 s was generous then: the gate ran at whatever small batch the
test happened to type.

Since the gate began ENFORCING --batch, the work behind that wall is set by the caller.
Qwen-Image-Edit measures 16.06 s per scheduler step at B=4 and 120 s at B=32, so the same 50-step
schedule that finishes in ~15 min at B=4 needs ~100 min at B=32 -- and the flat wall kills a HEALTHY
run mid-gate, reporting "exceeded 2700s with no verdict (likely device/fabric hang)": a hardware
verdict on hardware that was fine, after hours of builder work.

What is pinned: B=1 is bit-identical to before, an operator's own value stays absolute, and the
caller's budget still bounds everything.
"""

from __future__ import annotations

import inspect

import pytest

from scripts.tt_hw_planner.commands import emit_e2e as E

WALL = E._gate_wall_s
ENV = E._GATE_WALL_ENV


def test_batch_one_is_exactly_what_it_always_was(monkeypatch):
    """The whole point of scaling: the unscaled case must not move."""
    monkeypatch.delenv(ENV, raising=False)
    assert WALL(1) == 2700


def test_the_wall_grows_with_the_batch_the_gate_was_told_to_run(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert WALL(4) == 2700 * 4
    assert WALL(32) == 2700 * 32
    assert WALL(32) > WALL(4) > WALL(1)


def test_a_b32_gate_is_no_longer_killed_at_45_minutes(monkeypatch):
    """~100 min is the measured need at B=32; the old flat wall was 45 min."""
    monkeypatch.delenv(ENV, raising=False)
    assert WALL(32) > 100 * 60


@pytest.mark.parametrize("bad", [None, 0, -5, "x", ""])
def test_an_unusable_batch_falls_back_to_the_unscaled_wall(monkeypatch, bad):
    """Never smaller than it was, never an exception, whatever arrives."""
    monkeypatch.delenv(ENV, raising=False)
    assert WALL(bad) == 2700


def test_an_operator_value_stays_absolute(monkeypatch):
    """The variable's meaning must not change under someone who already sets it: asking for a
    tight wall must GIVE a tight wall, not that times the batch."""
    monkeypatch.setenv(ENV, "900")
    assert WALL(1) == 900
    assert WALL(32) == 900


def test_an_unparseable_override_does_not_crash_the_gate(monkeypatch):
    monkeypatch.setenv(ENV, "not-a-number")
    assert WALL(32) == 2700 * 32


def test_the_callers_budget_still_bounds_it():
    """min(timeout_s, wall) -- scaling can tighten toward the caller's budget, never exceed it."""
    src = inspect.getsource(E._run_deterministic_gates)
    assert "min(int(timeout_s), _gate_wall_s(batch))" in src


def test_the_wall_is_derived_not_typed_at_the_call_site():
    """No second literal: the call site asks the helper, it does not spell a number."""
    src = inspect.getsource(E._run_deterministic_gates)
    assert "2700" not in src, "the wall was re-typed at the call site"
    assert 'os.environ.get("E2E_GATE_HANG_TIMEOUT"' not in src, "the env name belongs to the helper"
