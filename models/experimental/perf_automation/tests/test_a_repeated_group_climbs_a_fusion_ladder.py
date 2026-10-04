"""A group of ops repeated within a layer climbs a fusion ladder instead of closing after the fold.

The fold gate offered one fix -- concatenate the repeats into one wider op -- and after three tries
the group closed, so a group the fold could not help never reached the step that could. Qwen-Image-
Edit's precise matmuls (the same weight times the bf16 hi and lo parts of one input, summed in fp32)
were folded where possible and never offered one kernel for the group. The ladder: fold, share,
fuse-ttnn, fuse-cpp, fuse-tt-lang -- each only once the one before is spent without a win. Op names
here are this file's own.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]


@pytest.fixture
def m(monkeypatch):
    spec = importlib.util.spec_from_file_location("pm_fusion_ladder_ut", str(_PA / "cc_optimize" / "perf_mcp.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_ledger", lambda: type("L", (), {"is_win": staticmethod(lambda a: bool(a.get("won")))})())
    monkeypatch.setattr(mod, "_load_attempts", lambda: [])
    monkeypatch.setattr(mod, "_ttl_available", lambda: True)
    for _kinds, cap_env in mod._GATE_LEVERS:
        monkeypatch.delenv(cap_env, raising=False)
    return mod


def _prof():
    def op(code, count, gap):
        return {"op_code": code, "count": count, "gap_ms": gap, "bucket": "matmul"}

    return {"device_ms": 100.0, "open_ops": [op("A", 30, 1.0), op("B", 30, 1.0), op("Pair", 90, 5.0)]}


def _rung(m, attempts):
    b = m._fold_gate(_prof(), attempts)
    return b and b["next_rung"]


def test_the_ladder_is_walked_in_order_one_rung_at_a_time(m):
    seen, spent = [], []
    while True:
        r = _rung(m, spent)
        if r is None:
            break
        seen.append(r)
        kind = {"structural-fold": "fold"}.get(r, r.replace("structural-", ""))
        spent.append({"kernel_kind": kind})
        assert len(spent) < 20, "the ladder never closes"
    assert seen == ["structural-fold"] * 3 + ["structural-share", "structural-fuse-ttnn", "fuse-cpp", "fuse-tt-lang"]


def test_a_win_on_any_rung_answers_the_group(m):
    for kind in ("fold", "share", "fuse-ttnn", "fuse-cpp", "fuse-tt-lang"):
        assert _rung(m, [{"kernel_kind": kind, "won": True}]) is None, kind


def test_a_rung_nobody_can_climb_does_not_hold_the_group(m, monkeypatch):
    monkeypatch.setattr(m, "_ttl_available", lambda: False)
    spent = [{"kernel_kind": "fold"}] * 3 + [{"kernel_kind": k} for k in ("share", "fuse-ttnn", "fuse-cpp")]
    assert _rung(m, spent) is None


def test_the_new_rungs_take_one_try_each_and_the_fold_keeps_three(m, monkeypatch):
    assert m._gate_cap(m._FOLD_LEVER[1]) == 3
    for _kinds, cap_env in m._FUSION_LADDER[1:]:
        assert m._gate_cap(cap_env) == 1
        monkeypatch.setenv(cap_env, "2")
        assert m._gate_cap(cap_env) == 2, "overridable like every gate cap"


def test_the_kernel_rungs_must_show_a_kernel_and_the_others_need_none(m):
    for kind in ("fuse-cpp", "fuse-tt-lang"):
        assert kind in m._KERNEL_AUTHORED_RUNGS and kind not in m._GATE_KINDS
    for kind in ("share", "structural-share", "fuse-ttnn", "structural-fuse-ttnn"):
        assert kind in m._GATE_KINDS


def test_each_rung_names_its_own_kind_and_its_guide_exists(m):
    from agent import router

    for lever in m._FUSION_LADDER[1:]:
        t = m._fusion_rung_target(lever, {"op_code": "Pair", "count": 90, "bucket": "matmul"}, 30, 5.0)
        assert "'%s'" % lever[0][0] in t["reason"] and t["next_rung"] == lever[0][-1]
        assert "3x per layer" in t["reason"] and "none: <why not>" in t["reason"]
    assert "fuse-group-kernel" in m._FUSION_RUNG_ASKS["fuse-cpp"]
    assert "One kernel for a repeated op group" in router.read_section("fuse-group-kernel", _PA / "GUIDELINES")
    assert any(
        e["id"] == "fuse-group-kernel" and "matmul" in e["op_class"] for e in router.build_index(_PA / "GUIDELINES")
    )


def test_the_recorder_permits_each_rung_its_cap(m):
    for kinds, cap_env in m._FUSION_LADDER:
        for kind in kinds:
            _tries, allowed = m._rung_allowance("Pair", kind, [])
            assert allowed >= m._gate_cap(cap_env), kind


def test_a_none_answer_on_a_kernel_rung_counts_and_a_kernelless_claim_does_not(m):
    ok = m._attempt_counts_for_the_ladder
    assert ok({"kernel_kind": "fuse-cpp", "note": "none: the parts use different weights"})
    assert not ok({"kernel_kind": "fuse-cpp", "note": "fused it, 5% faster"}), "a claim needs a kernel"
    assert ok({"kernel_kind": "fuse-cpp", "kernel_detected_in_source": True})
    assert not ok({"kernel_kind": "cpp", "note": "none: nothing to write"}), "single-op kernel rungs unchanged"
    assert ok({"kernel_kind": "share"}) and ok({"kernel_kind": "grid"})


def test_the_stop_gate_filters_attempts_through_that_one_rule():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    assert "attempts = [a for a in _load_attempts_all() if _attempt_counts_for_the_ladder(a)]" in src


def test_the_gates_that_ride_with_the_fold_still_ride_only_with_it():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("fold_block = _fold_gate(_gate_prof, attempts)")
    seg = src[i : i + 600]
    assert 'if fold_block and fold_block.get("next_rung") == _FOLD_LEVER[0][-1]:' in seg
    assert seg.index("_FOLD_LEVER[0][-1]") < seg.index("_order_gate(_gate_prof, attempts)")
