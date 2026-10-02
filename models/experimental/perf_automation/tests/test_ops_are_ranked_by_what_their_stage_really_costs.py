"""An op's gap is weighed by what its stage costs in the real pipeline, not in the capped profile.

The ranking reads op gaps off the profile, and the profile is capped: a stack the depth knob cuts
runs 2 blocks, a stack it cannot cut runs in full. Qwen-Image-Edit 2026-10-02: the denoiser was 31%
of the profile and ~49% of the full pipeline, the VAE 18.5% and ~1.4%. A weight per stage -- its
full-pipeline time over its profiled time, normalised to mean 1 -- puts the stages back on one scale.
Stage names here are this file's own.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_CC = Path(__file__).resolve().parents[1] / "cc_optimize"


@pytest.fixture
def pm(monkeypatch):
    spec = importlib.util.spec_from_file_location("pm_weights_ut", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "_model_key", lambda: "m")
    return m


def _profile(**stage_ms):
    return {"stage_buckets": {s: [{"device_ms": ms / 2}, {"device_ms": ms / 2}] for s, ms in stage_ms.items()}}


def test_an_undersampled_stage_weighs_more(pm, monkeypatch):
    monkeypatch.setattr(pm, "read_stage_ms", lambda **k: {"cut": 1600.0, "uncut": 30.0})
    w = pm.stage_cost_weights(_profile(cut=100.0, uncut=30.0))
    assert w["cut"] > 1.0 > w["uncut"]
    assert sum(w.values()) / len(w) == pytest.approx(1.0), "mean 1: the units stay the profile's"
    assert w["cut"] / w["uncut"] == pytest.approx(16.0)


@pytest.mark.parametrize(
    "full,prof",
    [
        ({}, {"a": 10.0, "b": 20.0}),  # no replay yet
        ({"a": 5.0, "b": 9.0}, {}),  # an unmarked capture
        ({"a": 5.0}, {"a": 10.0}),  # one stage has nothing to be weighed against
        ({"a": 5.0, "b": 9.0}, {"a": 10.0, "b": 0.0}),  # a stage with no profiled time
    ],
)
def test_no_weights_without_both_readings_for_two_stages(pm, monkeypatch, full, prof):
    monkeypatch.setattr(pm, "read_stage_ms", lambda **k: full)
    assert pm.stage_cost_weights(_profile(**prof)) == {}


def test_a_reader_that_fails_leaves_the_ranking_unweighted(pm, monkeypatch):
    def _boom(**k):
        raise OSError("state dir gone")

    monkeypatch.setattr(pm, "read_stage_ms", _boom)
    assert pm.stage_cost_weights(_profile(a=1.0, b=2.0)) == {}


def test_the_weight_reorders_and_its_absence_does_not(pm):
    rows = [
        {"op": "big_in_oversampled", "stage": "x", "eff_gap_ms": 100.0, "cost_weight": 0.5},
        {"op": "mid_in_undersampled", "stage": "y", "eff_gap_ms": 60.0, "cost_weight": 2.0},
        {"op": "unplaced", "stage": "", "eff_gap_ms": 80.0},
    ]
    order = [r["op"] for r in sorted(rows, key=lambda b: pm._blocking_order_key(b, set()))]
    assert order == ["mid_in_undersampled", "unplaced", "big_in_oversampled"]
    plain = [dict(r, cost_weight=1.0) for r in rows]
    order = [r["op"] for r in sorted(plain, key=lambda b: pm._blocking_order_key(b, set()))]
    assert order == ["big_in_oversampled", "unplaced", "mid_in_undersampled"], "weight 1 is the old order"


def test_a_finished_stage_still_goes_last_whatever_its_weight(pm):
    rows = [
        {"op": "finished_heavy", "stage": "done", "eff_gap_ms": 50.0, "cost_weight": 10.0},
        {"op": "short_light", "stage": "short", "eff_gap_ms": 1.0, "cost_weight": 0.1},
    ]
    order = [r["op"] for r in sorted(rows, key=lambda b: pm._blocking_order_key(b, {"short"}))]
    assert order == ["short_light", "finished_heavy"]


def test_the_gate_weighs_before_it_sorts_and_reports_the_gap_unchanged():
    src = (_CC / "perf_mcp.py").read_text()
    i = src.index("_weights = stage_cost_weights(prof)")
    j = src.index("blocking.sort(key=lambda b: _blocking_order_key(b, _short_names))")
    assert i < j
    assert 'b["cost_weight"] = round(float(_weights.get(b.get("stage") or "", 1.0)), 4)' in src[i:j]
    assert src.count("blocking.sort(") == 1, "one ordering"
    k = src.index("    next_target = (")
    assert '"gap_ms": blocking[0]["gap_ms"],' in src[k : k + 600], "the target reports the measured gap"
