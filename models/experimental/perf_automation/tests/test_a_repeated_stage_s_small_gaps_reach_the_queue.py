"""An op's gap is judged material by what it costs ONE REQUEST, not one profiled pass.

termination_check dropped every op whose profiled gap was under the material cutoff, and only then
weighed the survivors by their stage's cost. A stage a request repeats N times shows each op once in
the profile, so every op of that stage fell under the cutoff before the xN could apply: the queue
held only stages that run once, and the stage the band check named as still short was never offered
an op. Stage names, op codes and shapes here are this file's own.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]
_CC = _PA / "cc_optimize"

_STEP_SHAPE = "1x64x64"
_PASS_SHAPE = "128x64x64"


@pytest.fixture
def pm(monkeypatch, tmp_path):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("PERF_MCP_MATERIAL_GAP_MS", raising=False)
    spec = importlib.util.spec_from_file_location("pm_per_request_ut", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "_model_key", lambda: "m")
    monkeypatch.setattr(m, "read_stage_ms", lambda **k: {})
    monkeypatch.setattr(m, "read_stage_spread", lambda **k: {})
    return m


def _op(code, shape, gap, device_ms=None, **kw):
    return {
        "op_code": code,
        "shape": shape,
        "gap_ms": gap,
        "eff_gap_ms": gap,
        "device_ms": device_ms if device_ms is not None else gap * 2,
        "count": kw.pop("count", 2),
        **kw,
    }


def _profile(step_ops, pass_ops, device_ms=40.0):
    """A profile with two marked stages: `step` (a request repeats it) and `pass` (runs once)."""

    def _bucket(ops):
        return [{"id": "b", "device_ms": sum(o["device_ms"] for o in ops), "top_ops": [dict(o) for o in ops]}]

    return {
        "device_ms": device_ms,
        "stage_buckets": {"step": _bucket(step_ops), "pass": _bucket(pass_ops)},
        "open_ops": step_ops + pass_ops,
    }


def _run_termination_check(pm, monkeypatch, step_ops, pass_ops, spread=None):
    """The real termination_check over one profile, the device and the other gates stubbed out."""
    prof = _profile(step_ops, pass_ops)
    open_ops = [dict(o, at_floor=False) for o in step_ops + pass_ops]
    monkeypatch.setattr(pm, "_profile_with_zero_row_retry", lambda *a, **k: dict(prof))
    monkeypatch.setattr(pm.roofline, "residual_report", lambda p, env: {"open_ops": open_ops, "at_floor": False})
    monkeypatch.setattr(pm, "_persist_throughput", lambda *a, **k: None)
    monkeypatch.setattr(pm, "_load_attempts_all", lambda: [])
    if spread is not None:
        monkeypatch.setattr(pm, "read_stage_spread", lambda **k: spread)
    for gate in ("_host_gate", "_decode_gate", "_conv_gate", "_fold_gate"):
        monkeypatch.setattr(pm, gate, lambda *a, **k: None)
    monkeypatch.setattr(pm, "_short_stage_names", lambda: {"step"})
    return pm.termination_check()


def _repeats(pm, monkeypatch, reps):
    monkeypatch.setattr(pm, "stage_repeats_per_request", lambda stages: {s: reps[s] for s in stages or () if s in reps})


def test_a_repeated_stage_s_sub_cutoff_gap_is_material_per_request(pm, monkeypatch):
    _repeats(pm, monkeypatch, {"step": 127})
    small = _op("Matmul", _STEP_SHAPE, 0.02)
    prof = _profile([small], [_op("Matmul", _PASS_SHAPE, 3.0)])
    rd = pm._per_request_reader(prof)
    assert pm._material_gap_ms(prof["device_ms"]) == pytest.approx(0.25), "the cutoff the raw gap fails"
    assert rd.stage(small) == "step", "placed by fingerprint, not by the class name both stages share"
    assert rd.gap(small, 0.02) == pytest.approx(0.02 * 127)
    assert rd.material(small, 0.02)


def test_the_same_gap_in_a_stage_that_runs_once_is_still_not_material(pm, monkeypatch):
    _repeats(pm, monkeypatch, {"step": 127})
    once = _op("Softmax", _PASS_SHAPE, 0.02)
    prof = _profile([_op("Matmul", _STEP_SHAPE, 0.02)], [once])
    assert not pm._per_request_reader(prof).material(once, 0.02)


def test_membership_only_widens_an_op_that_was_material_stays_material(pm, monkeypatch):
    # sampling weights below 1 must never DROP an op the old test kept: dropping work is how a run stops
    monkeypatch.setattr(pm, "read_stage_ms", lambda **k: {"step": 100.0, "pass": 1.0})
    big = _op("Matmul", _PASS_SHAPE, 0.3)
    prof = _profile([_op("Matmul", _STEP_SHAPE, 0.02)], [big])
    rd = pm._per_request_reader(prof)
    assert rd.weights["pass"] < 1.0
    assert rd.material(big, 0.3)


def test_no_weights_is_exactly_the_old_test(pm):
    prof = _profile([_op("Matmul", _STEP_SHAPE, 0.02)], [_op("Matmul", _PASS_SHAPE, 0.3)])
    rd = pm._per_request_reader(prof)
    assert rd.weights == {}
    cut = pm._material_gap_ms(prof["device_ms"])
    for raw in (0.0, 0.02, cut - 1e-6, cut, 0.3):
        assert rd.material({}, raw) == (raw >= cut)
    assert rd.stage(prof["open_ops"][0]) == "", "unweighted: no placement is needed or done"


def test_a_gap_inside_its_stage_s_wobble_is_not_multiplied(pm, monkeypatch):
    _repeats(pm, monkeypatch, {"step": 127})
    noisy = _op("Matmul", _STEP_SHAPE, 0.02, device_ms=1.0)  # 0.02 <= 1.0 ms x 5%
    clear = _op("Add", _STEP_SHAPE, 0.08, device_ms=1.0)  # 0.08 > 0.05
    prof = _profile([noisy, clear], [_op("Matmul", _PASS_SHAPE, 3.0)])
    monkeypatch.setattr(pm, "read_stage_spread", lambda **k: {"step": 0.05})
    rd = pm._per_request_reader(prof)
    assert rd.gap(noisy, 0.02) == pytest.approx(0.02), "weight withheld: the stage cannot resolve it"
    assert not rd.material(noisy, 0.02)
    assert rd.gap(clear, 0.08) == pytest.approx(0.08 * 127)
    assert rd.material(clear, 0.08)


def test_the_per_request_cutoff_rises_with_the_units_and_never_falls(pm, monkeypatch):
    # a small profile: the raw cutoff sits under the 0.25 cap, the per-request one is restated in the
    # larger units (the step stage counted 127 times) and reaches the cap
    _repeats(pm, monkeypatch, {"step": 127})
    tiny = _op("Matmul", _STEP_SHAPE, 0.0015, device_ms=0.05)
    prof = _profile([tiny], [_op("Matmul", _PASS_SHAPE, 0.5, device_ms=1.0)], device_ms=2.0)
    assert pm._material_gap_ms(2.0) == pytest.approx(0.06)
    rd = pm._per_request_reader(prof)
    assert rd.gap(tiny, 0.0015) == pytest.approx(0.1905)
    assert not rd.material(tiny, 0.0015), "0.19 per request is under the per-request cutoff of 0.25"


def test_the_order_weighs_a_gap_as_its_cutoff_did(pm, monkeypatch):
    """One weight for both: a gap the cutoff held to its raw value (inside its stage's wobble) is not
    multiplied when the queue is ordered either, so the noise in it cannot jump the queue."""
    _repeats(pm, monkeypatch, {"step": 127})
    noisy = _op("Matmul", _STEP_SHAPE, 0.3, device_ms=10.0)  # material raw, but 0.3 <= 10.0 ms x 5%
    clear = _op("Add", _STEP_SHAPE, 0.08, device_ms=1.0)
    prof = _profile([noisy, clear], [_op("Matmul", _PASS_SHAPE, 3.0)])
    monkeypatch.setattr(pm, "read_stage_spread", lambda **k: {"step": 0.05})
    rd = pm._per_request_reader(prof)
    assert rd.material(noisy, 0.3), "kept: the raw gap clears the old cutoff"
    assert rd.weight(noisy, 0.3) == 1.0 and rd.weight(clear, 0.08) == 127.0
    assert rd.gap(clear, 0.08) == pytest.approx(0.08 * rd.weight(clear, 0.08)), "gap is raw x weight"


def test_termination_check_orders_by_the_weight_its_cutoff_used(pm, monkeypatch):
    _repeats(pm, monkeypatch, {"step": 127})
    noisy = _op("Matmul", _STEP_SHAPE, 0.3, device_ms=10.0, bound_by="memory", bucket="matmul")
    clear = _op("Add", _STEP_SHAPE, 0.08, device_ms=1.0, bound_by="memory", bucket="eltwise")
    out = _run_termination_check(pm, monkeypatch, [noisy, clear], [], spread={"step": 0.05})
    blocking = out.get("blocking_ops") or out.get("blocking") or []
    weights = {b.get("op"): b.get("cost_weight") for b in blocking}
    assert weights == {"Matmul %s" % _STEP_SHAPE: 1.0, "Add %s" % _STEP_SHAPE: 127.0}, blocking
    assert blocking[0]["op"] == "Add %s" % _STEP_SHAPE, "10.2 ms per request outranks a raw 0.3 ms"


def test_a_shared_class_name_with_no_fingerprint_match_is_not_placed(pm, monkeypatch):
    _repeats(pm, monkeypatch, {"step": 127})
    prof = _profile([_op("Matmul", _STEP_SHAPE, 0.02)], [_op("Matmul", _PASS_SHAPE, 3.0)])
    stray = _op("Matmul", "9x9x9", 0.02)
    prof["open_ops"].append(stray)
    rd = pm._per_request_reader(prof)
    assert rd.stage(stray) == "", "three open Matmul groups: the class name cannot place this one"
    assert not rd.material(stray, 0.02)


def test_a_gate_sees_a_repeated_stage_s_group(pm, monkeypatch):
    _repeats(pm, monkeypatch, {"step": 127})
    convs = [_op("Conv2d", _STEP_SHAPE, 0.01, bucket="conv_pool")]
    prof = _profile(convs, [_op("Matmul", _PASS_SHAPE, 3.0)])
    assert pm._per_request_reader(prof).material_sum(convs)
    _repeats(pm, monkeypatch, {})
    assert not pm._per_request_reader(prof).material_sum(convs), "unweighted: the old raw test"


def test_every_gate_judges_materiality_through_the_one_reader():
    src = (_CC / "perf_mcp.py").read_text()
    for gate in ("_conv_gate", "_fold_gate", "_order_gate", "_split_gate"):
        i = src.index("def %s(" % gate)
        body = src[i : src.index("\ndef ", i + 1)]
        assert "_per_request_reader(prof)" in body, gate
        assert "_material_gap_ms(" not in body, gate
    i = src.index("def termination_check(")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "_req.material(o, max(gap, eff_gap))" in body
    assert body.index("_per_request_reader(") < body.index('for o in rep.get("open_ops")')


def test_a_step_delta_states_what_it_is_worth_to_a_request(pm, monkeypatch):
    _repeats(pm, monkeypatch, {"step": 127})
    rows = pm._stage_deltas({"step": 1.93, "pass": 50.0}, {"step": 1.95, "pass": 50.0}, {"step": 0.001})
    assert rows["step"]["repeats"] == 127
    assert rows["step"]["delta_per_request_ms"] == pytest.approx(-0.02 * 127, abs=1e-3)
    assert rows["step"]["improved"] is True, "the per-step delta is still what is judged"
    assert "delta_per_request_ms" not in rows["pass"], "a stage that runs once states nothing extra"


def test_termination_check_queues_a_repeated_stage_s_ops(pm, monkeypatch):
    """The real termination_check, fed one profile: the step stage's op reaches `blocking`."""
    _repeats(pm, monkeypatch, {"step": 127})
    step = _op("Matmul", _STEP_SHAPE, 0.02, bound_by="memory", bucket="matmul")
    once = _op("Matmul", _PASS_SHAPE, 0.3, bound_by="memory", bucket="matmul")
    out = _run_termination_check(pm, monkeypatch, [step], [once])
    blocking = out.get("blocking_ops") or out.get("blocking") or []
    names = [(b.get("stage"), b.get("op")) for b in blocking]
    step_name = "Matmul %s" % _STEP_SHAPE  # an op_code with no dims is named by its shape
    assert ("step", step_name) in names, out
    assert names[0] == ("step", step_name), "the short, repeated stage is worked first"
    assert out["can_stop"] is False
