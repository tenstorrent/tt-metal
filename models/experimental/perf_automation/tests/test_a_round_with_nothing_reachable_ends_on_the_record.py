# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The gate said "call finish_round"; finish_round said "call termination_check".

Devstral run 4, 2026-10-18, round 1: decode measured 17.99 ms against a 10.45 ms band, every material
op's rungs already tried or crash-closed, so termination_check returned can_stop=false with no
next_target and the directive "nothing to work this round -- call finish_round so the round ends on
the record". finish_round then refused with "Go back to the top of the LOOP: call termination_check,
take the next target" -- a target that did not exist. The agent, handed two instructions that pointed
at each other, went looking through the tool's source for a way out.

The refusal stands: a band that cannot be reached is not waived, and the run ends on its round budget
as designed. What changes is what the agent is told to do in that one state -- exit -- and the gate's
directive for the same state says the same. Op codes, shapes and stage names here are this file's own.
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
    spec = importlib.util.spec_from_file_location("pm_nothing_reachable_ut", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "_model_key", lambda: "m")
    monkeypatch.setattr(m, "_load_attempts_all", lambda: [])
    return m


_SHORT = [{"stage": "pass", "measured_ms": 17.99, "achievable_ms": 10.45, "over_by_ms": 7.54, "binds": "memory"}]


def _finish(pm, monkeypatch, *, short, gate, untried=()):
    """finish_round with the gate's answer, the short list and the untried list all stated."""
    recorded = []
    monkeypatch.setattr(pm, "_stages_short_of_achievable", lambda: list(short))
    monkeypatch.setattr(pm, "termination_check", gate)
    monkeypatch.setattr(pm, "_untried_material_ops", lambda blocking, attempts: list(untried))
    monkeypatch.setattr(pm, "_record_round_finish", lambda v: recorded.append(dict(v)))
    out = pm.finish_round()
    return out, recorded


def test_a_short_stack_with_no_target_is_refused_and_told_to_exit(pm, monkeypatch):
    out, rec = _finish(
        pm, monkeypatch, short=_SHORT, gate=lambda: {"can_stop": False, "next_target": None, "blocking_ops": []}
    )
    assert out["finished"] is False, "a band that cannot be reached is not waived"
    assert out["nothing_reachable"] is True
    assert "NOT FINISHED" in out["why"] and "exit now" in out["why"], out["why"]
    assert "Go back to the top of the LOOP" not in out["why"], "sent back to a loop with no target"
    assert "pass" in out["why"]
    assert rec and rec[-1].get("nothing_reachable") is True, "the refusal is on the record"


def test_a_short_stack_with_a_target_is_still_sent_back_to_the_loop(pm, monkeypatch):
    target = {"op": "Softmax", "shape": "1x32x32", "stage": "pass", "rung": "grid"}
    out, _ = _finish(
        pm, monkeypatch, short=_SHORT, gate=lambda: {"can_stop": False, "next_target": target, "blocking_ops": [target]}
    )
    assert out["finished"] is False
    assert "Go back to the top of the LOOP" in out["why"]
    assert "nothing_reachable" not in out


def test_an_unworked_rung_is_still_sent_back_to_the_loop(pm, monkeypatch):
    left = [{"op": "Softmax", "shape": "1x32x32", "rung": "grid"}]
    out, _ = _finish(
        pm,
        monkeypatch,
        short=_SHORT,
        gate=lambda: {"can_stop": False, "next_target": None, "blocking_ops": left},
        untried=left,
    )
    assert out["finished"] is False
    assert "Go back to the top of the LOOP" in out["why"]
    assert "nothing_reachable" not in out


def test_a_gate_that_could_not_answer_does_not_claim_nothing_is_reachable(pm, monkeypatch):
    """An empty gate answer is not a gate answer of 'no target'."""

    def _boom():
        raise RuntimeError("profile failed")

    out, _ = _finish(pm, monkeypatch, short=_SHORT, gate=_boom)
    assert out["finished"] is False
    assert "nothing_reachable" not in out
    assert "Go back to the top of the LOOP" in out["why"]


def test_every_stack_in_band_and_nothing_left_still_finishes(pm, monkeypatch):
    out, rec = _finish(
        pm, monkeypatch, short=[], gate=lambda: {"can_stop": True, "next_target": None, "blocking_ops": []}
    )
    assert out["finished"] is True
    assert rec and rec[-1]["finished"] is True


def test_the_gate_and_finish_round_agree_on_what_to_do_when_nothing_is_reachable(pm, monkeypatch):
    """THE SEAM. The gate's directive for this state names finish_round and says exit; finish_round's
    refusal for the same state says exit and does not name termination_check as the next move."""
    op = {
        "op_code": "Softmax",
        "shape": "1x32x32",
        "gap_ms": 0.001,
        "eff_gap_ms": 0.001,
        "device_ms": 0.01,
        "count": 1,
        "bound_by": "memory",
        "bucket": "other",
    }
    prof = {
        "device_ms": 40.0,
        "stage_buckets": {"pass": [{"id": "b", "device_ms": op["device_ms"], "top_ops": [dict(op)]}]},
        "open_ops": [dict(op)],
    }
    monkeypatch.setattr(pm, "read_stage_ms", lambda **k: {})
    monkeypatch.setattr(pm, "read_stage_spread", lambda **k: {})
    monkeypatch.setattr(pm, "_profile_with_zero_row_retry", lambda *a, **k: dict(prof))
    monkeypatch.setattr(pm.roofline, "residual_report", lambda p, env: {"open_ops": [dict(op, at_floor=False)]})
    monkeypatch.setattr(pm, "_persist_throughput", lambda *a, **k: None)
    for gate in ("_host_gate", "_decode_gate", "_conv_gate", "_fold_gate"):
        monkeypatch.setattr(pm, gate, lambda *a, **k: None)
    monkeypatch.setattr(pm, "_stages_short_of_achievable", lambda: list(_SHORT))
    verdict = pm.termination_check()
    assert verdict["can_stop"] is False and verdict["next_target"] is None
    assert "finish_round" in verdict["directive"] and "exit" in verdict["directive"]
    assert "work next_target" not in verdict["directive"]

    monkeypatch.setattr(pm, "_record_round_finish", lambda v: None)
    out = pm.finish_round()  # the real gate, the real untried computation (nothing blocks)
    assert out["finished"] is False and out["nothing_reachable"] is True
    assert "exit now" in out["why"] and "Go back to the top of the LOOP" not in out["why"]
