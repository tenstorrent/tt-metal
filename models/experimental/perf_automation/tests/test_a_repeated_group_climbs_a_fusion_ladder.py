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
        spent.append({"kernel_kind": kind, "op_signature": "Pair"})
        assert len(spent) < 20, "the ladder never closes"
    assert seen == ["structural-fold"] * 3 + ["structural-share", "structural-fuse-ttnn", "fuse-cpp", "fuse-tt-lang"]


def test_a_win_on_any_rung_answers_the_group(m):
    for kind in ("fold", "share", "fuse-ttnn", "fuse-cpp", "fuse-tt-lang"):
        assert _rung(m, [{"kernel_kind": kind, "won": True, "op_signature": "Pair"}]) is None, kind


def test_a_rung_nobody_can_climb_does_not_hold_the_group(m, monkeypatch):
    monkeypatch.setattr(m, "_ttl_available", lambda: False)
    spent = [{"kernel_kind": "fold", "op_signature": "Pair"}] * 3 + [
        {"kernel_kind": k, "op_signature": "Pair"} for k in ("share", "fuse-ttnn", "fuse-cpp")
    ]
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


# --------------------------------------------------------------------------------------------------
# the ladder is PER GROUP: one group's answer does not answer another's
# --------------------------------------------------------------------------------------------------


def _two_groups():
    def op(code, count, gap):
        return {"op_code": code, "count": count, "gap_ms": gap, "bucket": "matmul"}

    # a reduction that was folded long ago, and a matmul group of the same per-layer multiple
    return {
        "device_ms": 100.0,
        "open_ops": [op("A", 30, 1.0), op("B", 30, 1.0), op("Reduce", 90, 6.0), op("Mm 64 x 64 x 64", 90, 5.0)],
    }


def test_a_win_on_one_group_leaves_another_group_on_its_own_rung(m):
    """The defect: an early fold win on one group closed the ladder for every group of the model."""
    won_elsewhere = [{"kernel_kind": "fold", "won": True, "op_signature": "Reduce"}]
    b = m._fold_gate(_two_groups(), won_elsewhere)
    assert b and b["op"] == "Mm 64 x 64 x 64" and b["next_rung"] == "structural-fold"


def test_the_biggest_open_group_is_worked_first(m):
    b = m._fold_gate(_two_groups(), [])
    assert b["op"] == "Reduce", "biggest gap first, as before"


def test_a_group_past_its_fold_reaches_the_fusion_rungs_on_its_own_history(m):
    hist = [{"kernel_kind": "fold", "won": True, "op_signature": "Reduce"}]
    hist += [{"kernel_kind": "fold", "op_signature": "Mm 64 x 64 x 64"}] * 3
    assert m._fold_gate(_two_groups(), hist)["next_rung"] == "structural-share"


def test_an_attempt_on_another_shape_is_not_this_group_s_attempt(m):
    """_op_match's own rule: a shaped op matches only the same shape."""
    other_shape = [{"kernel_kind": "fold", "won": True, "op_signature": "Mm 64 x 64 x 32"}]
    other_shape += [{"kernel_kind": "fold", "won": True, "op_signature": "Reduce"}]
    b = m._fold_gate(_two_groups(), other_shape)
    assert b and b["op"] == "Mm 64 x 64 x 64" and b["next_rung"] == "structural-fold"


def test_every_group_answered_closes_the_gate(m):
    done = [{"kernel_kind": "fold", "won": True, "op_signature": s} for s in ("Reduce", "Mm 64 x 64 x 64")]
    assert m._fold_gate(_two_groups(), done) is None


def test_the_gate_scopes_with_the_per_op_ladder_s_matcher():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("def _fold_gate(")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "_fusion_rung([a for a in attempts if _op_match(" in body
    assert "_fusion_rung(attempts)" not in body, "no model-wide reading of the ladder is left"


# --------------------------------------------------------------------------------------------------
# which groups are held: named by op AND shape, ranked by per-request cost, bounded
# --------------------------------------------------------------------------------------------------


def _staged(*groups):
    """A profile whose open ops are (op_code, shape, count, gap); eight once-per-layer ops set the mode."""

    def op(code, shape, count, gap):
        return {"op_code": code, "shape": shape, "count": count, "gap_ms": gap, "bucket": "x"}

    once = [op("Once%d" % i, "", 30, 1.0) for i in range(8)]
    return {"device_ms": 100.0, "open_ops": once + [op(*g) for g in groups]}


def test_a_shapeless_op_is_named_by_its_shape(m):
    assert m._group_name({"op_code": "Elt", "shape": "8x16 @ 8x16"}) == "Elt 8x16 @ 8x16"
    assert m._group_name({"op_code": "Mm 4 x 4 x 4", "shape": "4x4 @ 4x4"}) == "Mm 4 x 4 x 4", "dims already name it"
    assert m._group_name({"op_code": "Elt"}) == "Elt"


def test_a_win_on_one_shape_of_an_elementwise_class_is_not_another_shape_s(m):
    prof = _staged(("Elt", "8x16 @ 8x16", 90, 6.0), ("Elt", "8x32 @ 8x32", 90, 5.0))
    won = [{"kernel_kind": "fold", "won": True, "op_signature": "Elt 8x16 @ 8x16"}]
    b = m._fold_gate(prof, won)
    assert b and b["op"] == "Elt 8x32 @ 8x32" and b["next_rung"] == "structural-fold"


def test_a_group_in_a_repeated_stage_is_held_before_a_bigger_gap_that_runs_once(m, monkeypatch):
    prof = _staged(("Once 1 x 1 x 1", "", 90, 9.0), ("Loop 2 x 2 x 2", "", 90, 1.0))
    monkeypatch.setattr(m, "stage_cost_weights", lambda p: {"loop": 50.0, "once": 1.0})
    monkeypatch.setattr(
        m, "stage_of_op", lambda op, p: {"Once 1 x 1 x 1": "once", "Loop 2 x 2 x 2": "loop"}.get(op, "")
    )
    assert m._fold_gate(prof, [])["op"] == "Loop 2 x 2 x 2", "1 ms x 50 per request beats 9 ms x 1"


def test_a_class_name_shared_by_several_groups_borrows_no_stage(m, monkeypatch):
    """Several open groups under one bare name cannot be placed by name: they keep weight 1."""
    prof = _staged(("Elt", "8x16 @ 8x16", 90, 9.0), ("Elt", "8x32 @ 8x32", 90, 8.0), ("Mm 2 x 2 x 2", "", 90, 1.0))
    monkeypatch.setattr(m, "stage_cost_weights", lambda p: {"loop": 50.0})
    monkeypatch.setattr(m, "stage_of_op", lambda op, p: "loop")
    assert m._fold_gate(prof, [])["op"] == "Mm 2 x 2 x 2", "the shared name is not boosted to 450 / 400"


def test_only_the_held_groups_can_block(m, monkeypatch):
    groups = [("Mm %d x 1 x 1" % i, "", 90, 10.0 - i) for i in range(5)]
    done = [{"kernel_kind": "fold", "won": True, "op_signature": "Mm %d x 1 x 1" % i} for i in range(3)]
    monkeypatch.delenv(m._FUSION_GROUPS_ENV, raising=False)
    assert m._fold_gate(_staged(*groups), done) is None, "groups 4 and 5 are not held by default"
    monkeypatch.setenv(m._FUSION_GROUPS_ENV, "5")
    assert m._fold_gate(_staged(*groups), done)["op"] == "Mm 3 x 1 x 1"


def test_the_held_bound_caps_what_the_gate_can_owe(m):
    per_group = sum(m._gate_cap(c) for _k, c in m._FUSION_LADDER)
    assert int(m._FUSION_GROUPS_DEFAULT) * per_group <= 25, "a bounded number of answers ends the gate"
