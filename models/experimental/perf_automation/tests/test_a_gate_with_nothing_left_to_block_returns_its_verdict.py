"""The stop gate returns its verdict when nothing blocks, instead of crashing on an empty list.

termination_check set can_stop the moment the last material op cleared its checklist, then read
blocking[0] to carry the prior-attempt reason into next_target and raised IndexError. The harness
reads a raised gate as "not done" and launches another round, so a finished run relaunched empty
rounds until its round budget was spent (Devstral run 3, 2026-10-15: rounds 4-10 of 10 did nothing).
Op codes, shapes and stage names here are this file's own.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]
_CC = _PA / "cc_optimize"


@pytest.fixture
def pm(monkeypatch, tmp_path):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("pm_nothing_blocking_ut", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "_model_key", lambda: "m")
    monkeypatch.setattr(m, "read_stage_ms", lambda **k: {})
    monkeypatch.setattr(m, "read_stage_spread", lambda **k: {})
    return m


def _op(gap):
    return {
        "op_code": "Softmax",
        "shape": "1x32x32",
        "gap_ms": gap,
        "eff_gap_ms": gap,
        "device_ms": max(gap * 2, 0.01),
        "count": 1,
        "bound_by": "memory",
        "bucket": "other",
    }


def _gate(pm, monkeypatch, gap, short):
    """The real termination_check, fed a profile with ONE op of the given gap (cutoff: 0.25 ms)."""
    op = _op(gap)
    prof = {
        "device_ms": 40.0,
        "stage_buckets": {"pass": [{"id": "b", "device_ms": op["device_ms"], "top_ops": [dict(op)]}]},
        "open_ops": [dict(op)],
    }
    monkeypatch.setattr(pm, "_profile_with_zero_row_retry", lambda *a, **k: dict(prof))
    monkeypatch.setattr(pm.roofline, "residual_report", lambda p, env: {"open_ops": [dict(op, at_floor=False)]})
    monkeypatch.setattr(pm, "_persist_throughput", lambda *a, **k: None)
    monkeypatch.setattr(pm, "_load_attempts_all", lambda: [])
    for gate in ("_host_gate", "_decode_gate", "_conv_gate", "_fold_gate"):
        monkeypatch.setattr(pm, gate, lambda *a, **k: None)
    monkeypatch.setattr(pm, "_stages_short_of_achievable", lambda: list(short))
    return pm.termination_check()


def test_nothing_blocking_and_every_stack_in_band_is_a_clean_stop(pm, monkeypatch):
    out = _gate(pm, monkeypatch, gap=0.001, short=[])
    assert out["blocking_ops"] == [], out
    assert out["can_stop"] is True
    assert out["next_target"] is None
    assert out["directive"].startswith("DONE")


def test_nothing_blocking_while_a_stack_is_short_says_so_and_names_no_target(pm, monkeypatch):
    short = [{"stage": "pass", "measured_ms": 12.0, "over_by_ms": 2.0}]
    out = _gate(pm, monkeypatch, gap=0.001, short=short)
    assert out["blocking_ops"] == [], out
    assert out["can_stop"] is False
    assert out["next_target"] is None
    assert out["stages_short_of_achievable"] == short
    assert out["directive"].startswith("NOT DONE")
    assert "pass" in out["directive"], "the stack that still owes its band is named"
    assert "work next_target" not in out["directive"], "there is no target to work"
    assert "finish_round" in out["directive"]


def test_a_real_target_still_carries_the_prior_attempt_reason(pm, monkeypatch):
    monkeypatch.setattr(pm, "_last_recorded_note", lambda op: "wedged at grid 8x8")
    out = _gate(pm, monkeypatch, gap=0.3, short=[])
    assert out["can_stop"] is False
    assert out["next_target"]["op"] == "Softmax 1x32x32", out  # a shapeless op is named by its shape
    assert out["next_target"]["reason"].endswith("Last recorded attempt on this target: wedged at grid 8x8")
    assert "work next_target" in out["directive"]
