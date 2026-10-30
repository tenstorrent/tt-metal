"""Each shapeless open op climbs its own ladder: an attempt on one does not answer for another.

termination_check named every open op by its bare op_code. A matmul's op_code carries its dims, but a
shapeless class does not -- every custom kernel profiles as GenericOpDeviceOperation, every eltwise as
BinaryNgDeviceOperation -- and _op_match matches a shapeless name on class alone. So one attempt on any
instance answered the ladder for all of them. Observed 2026-10-19: one cpp try on a small attention
kernel closed the largest kernel of a prefill stage (a fused gate/up matmul, a third of the stage)
before any rung had been tried on it, and the run spent its remaining rounds with nothing to work.

The open op is now named by op and shape (_group_name, as the fold gate names its groups). The target
carries that name, so the agent records against it. Op codes, shapes and stage names here are this
file's own.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]
_CC = _PA / "cc_optimize"

_KERNEL = "CustomKernelOp"
_BIG = "4096x2048 @ 2048x12288"
_SMALL = "128x128 @ 128x128"


@pytest.fixture
def pm(monkeypatch, tmp_path):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("pm_shapeless_ladder_ut", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "_model_key", lambda: "m")
    monkeypatch.setattr(m, "read_stage_ms", lambda **k: {})
    monkeypatch.setattr(m, "read_stage_spread", lambda **k: {})
    return m


def _op(code, shape, gap):
    return {
        "op_code": code,
        "shape": shape,
        "gap_ms": gap,
        "eff_gap_ms": gap,
        "device_ms": gap * 2,
        "count": 2,
        "bound_by": "compute",
        "bucket": "other",
    }


def _gate(pm, monkeypatch, ops, attempts):
    """The real termination_check, fed these open ops and this attempt history."""
    prof = {
        "device_ms": 40.0,
        "stage_buckets": {"pass": [{"id": "b", "device_ms": 10.0, "top_ops": [dict(o) for o in ops]}]},
        "open_ops": [dict(o) for o in ops],
    }
    monkeypatch.setattr(pm, "_profile_with_zero_row_retry", lambda *a, **k: dict(prof))
    monkeypatch.setattr(
        pm.roofline, "residual_report", lambda p, env: {"open_ops": [dict(o, at_floor=False) for o in ops]}
    )
    monkeypatch.setattr(pm, "_persist_throughput", lambda *a, **k: None)
    monkeypatch.setattr(pm, "_load_attempts_all", lambda: list(attempts))
    monkeypatch.setattr(pm, "_load_attempts", lambda *a, **k: list(attempts))
    for gate in ("_host_gate", "_decode_gate", "_conv_gate", "_fold_gate"):
        monkeypatch.setattr(pm, gate, lambda *a, **k: None)
    monkeypatch.setattr(pm, "_stages_short_of_achievable", lambda: [])
    return pm.termination_check()


def _attempt(op, rung):
    """What the agent records for a target it worked: the target's own name and rung, no gain."""
    return {
        "op_signature": op,
        "kernel_kind": rung,
        "measured_ms": 10.0,
        "baseline_ms": 9.9,
        "beat_baseline": False,
        "kernel_detected_in_source": True,
        "pcc_status": "ok",
        "pcc": 0.999,
    }


def _names(out):
    return {b["op"] for b in out["blocking_ops"]}


def test_two_shapeless_ops_are_two_targets_named_by_their_shape(pm, monkeypatch):
    ops = [_op(_KERNEL, _BIG, 1.4), _op(_KERNEL, _SMALL, 0.35)]
    out = _gate(pm, monkeypatch, ops, attempts=[])
    assert _names(out) == {"%s %s" % (_KERNEL, _BIG), "%s %s" % (_KERNEL, _SMALL)}, out["blocking_ops"]
    assert out["next_target"]["op"] == "%s %s" % (_KERNEL, _BIG), "the costlier op is worked first"


def test_working_one_shapeless_op_through_its_ladder_leaves_the_other_untouched(pm, monkeypatch):
    big, small = "%s %s" % (_KERNEL, _BIG), "%s %s" % (_KERNEL, _SMALL)
    ops = [_op(_KERNEL, _SMALL, 1.4), _op(_KERNEL, _BIG, 0.35)]
    fresh = {b["op"]: b["next_rung"] for b in _gate(pm, monkeypatch, ops, attempts=[])["blocking_ops"]}
    attempts = []
    for _ in range(80):  # the agent works whatever the gate hands it, one rung per round
        out = _gate(pm, monkeypatch, ops, attempts)
        target = out["next_target"]
        if not target or target["op"] != small:
            break
        attempts.append(_attempt(target["op"], target["rung"]))
    else:
        pytest.fail("the small op's ladder never ran out")
    assert attempts, "the small op was worked"
    assert small not in _names(out), "the worked op cleared once its ladder was spent"
    blocking = {b["op"]: b for b in out["blocking_ops"]}
    assert big in blocking, "the other shapeless op is still open: no rung was ever tried on it"
    assert blocking[big]["next_rung"] == fresh[big], "its ladder starts where a fresh one does"
    assert out["next_target"]["op"] == big


def test_an_attempt_recorded_under_the_bare_class_name_answers_no_shaped_op(pm, monkeypatch):
    ops = [_op(_KERNEL, _BIG, 1.4)]
    fresh = _gate(pm, monkeypatch, ops, attempts=[])["blocking_ops"][0]["next_rung"]
    out = _gate(pm, monkeypatch, ops, attempts=[_attempt(_KERNEL, fresh)])
    assert out["blocking_ops"][0]["op"] == "%s %s" % (_KERNEL, _BIG)
    assert out["blocking_ops"][0]["next_rung"] == fresh, "a class-wide record cannot say which instance it tried"


def test_a_matmul_keeps_the_name_its_op_code_already_gives_it(pm, monkeypatch):
    code = "MatmulDeviceOperation 4096 x 2048 x 2048"
    out = _gate(pm, monkeypatch, [_op(code, "4096x2048 @ 2048x2048", 1.0)], attempts=[])
    assert out["next_target"]["op"] == code, "an op_code that carries its dims is already one group's name"


def test_a_shapeless_op_with_no_shape_on_record_keeps_its_bare_name(pm, monkeypatch):
    out = _gate(pm, monkeypatch, [_op(_KERNEL, "", 1.0)], attempts=[])
    assert out["next_target"]["op"] == _KERNEL
