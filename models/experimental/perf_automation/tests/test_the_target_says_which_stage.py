"""The next target must say which stage its op lives in, or the agent falls back to the metric.

next_target named an OP and nothing else, and the only stage-shaped signal the agent gets is the
metric in its prompt -- which describes the recurring stage. Handed a stage-less op, it reasoned
about the one stage it had been told to care about and kept returning there while the ranking
pointed elsewhere. Measured on voxtral mid-run: ~5 ms of headroom left in the recurring stage and
~195 ms sitting in the prompt-consuming one, with the agent's own words naming "the largest
remaining decode lever".
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
if str(PERF) not in sys.path:
    sys.path.insert(0, str(PERF))

_spec = _ilu.spec_from_file_location("_pm_stage", PERF / "cc_optimize" / "perf_mcp.py")
_pm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pm)

stage_of_op = _pm.stage_of_op
_stage_gap_share = _pm._stage_gap_share


def _prof(**stages):
    return {
        "stage_buckets": {
            name: [{"top_ops": [{"op_code": c, "device_ms": ms} for c, ms in ops]}] for name, ops in stages.items()
        }
    }


def test_an_op_is_attributed_to_where_it_costs_the_most_not_where_it_appears_first():
    """First-match returns whichever stage the capture serialised first -- a coin toss."""
    prof = _prof(first=[("Matmul", 1.0)], second=[("Matmul", 20.0)])
    assert stage_of_op("Matmul", prof) == "second"


def test_a_tie_names_no_stage():
    prof = _prof(one=[("X", 5.0)], two=[("X", 5.0)])
    assert stage_of_op("X", prof) == "", "a tie must say nothing rather than guess"


def test_an_op_the_capture_cannot_place_returns_nothing():
    """host_overhead is real and belongs to no stage; claiming one would be a lie."""
    prof = _prof(only=[("Matmul", 1.0)])
    assert stage_of_op("host_overhead", prof) == ""
    assert stage_of_op("", prof) == ""


def test_an_unmarked_capture_behaves_as_before_the_field_existed():
    for prof in ({}, {"stage_buckets": {}}, {"stage_buckets": None}, None):
        assert stage_of_op("Matmul", prof) == ""
        assert _stage_gap_share(prof) == {}


def test_the_stage_names_come_from_the_capture_not_from_a_list():
    """A model that calls its stages anything must still be attributed."""
    prof = _prof(denoise=[("Conv", 9.0)], upsample=[("Conv", 2.0)])
    assert stage_of_op("Conv", prof) == "denoise"


def test_the_share_summarises_every_marked_stage():
    prof = _prof(a=[("X", 1.5)], b=[("Y", 2.25)])
    assert _stage_gap_share(prof) == {"a": 1.5, "b": 2.25}


def test_the_target_carries_the_stage_and_the_directive_shows_where_time_is():
    """The stage reaches next_target. WHERE it is resolved is free to move -- and did: every
    candidate needs it now that a finished stage is ordered last, so it is resolved once onto the
    entry instead of again for whichever entry wins."""
    src = (PERF / "cc_optimize" / "perf_mcp.py").read_text(encoding="utf-8")
    assert 'entry["stage"] = stage_of_op(op_code, prof)' in src, "the entry must carry it"
    assert '"stage": blocking[0].get("stage")' in src, "next_target must pass it on"
    assert "IN THE STAGE " in src, "the directive must tell the agent to work that stage"
    assert "_stage_time_note" in src, "the directive must show where the time actually is"


def test_an_op_is_not_pooled_with_a_longer_op_that_starts_with_its_name():
    """A prefix test would attribute an op to the stage a DIFFERENT op is heaviest in.

    Real op codes nest: MorehSoftmaxOp is a prefix of MorehSoftmaxOpParallelizationStrategy. The
    target carries the bare code, so matching on a prefix pools the two and answers with the wrong
    stage -- the exact mis-steer this field exists to remove.
    """
    prof = _prof(
        cheap=[("MorehSoftmaxOp", 1.0)],
        expensive=[("MorehSoftmaxOpParallelizationStrategy", 40.0)],
    )
    assert stage_of_op("MorehSoftmaxOp", prof) == "cheap"
    assert stage_of_op("MorehSoftmaxOpParallelizationStrategy", prof) == "expensive"


def test_a_target_named_with_its_shape_lands_in_its_own_stage():
    """The same code runs in every stage; a target that carries the shape must not be widened."""
    prof = {
        "stage_buckets": {
            "prompt": [{"top_ops": [{"op_code": "Matmul", "shape": "[1,512,3072]", "device_ms": 21.7}]}],
            "step": [{"top_ops": [{"op_code": "Matmul", "shape": "[1,1,3072]", "device_ms": 2.4}]}],
        }
    }
    assert stage_of_op("Matmul [1,1,3072]", prof) == "step"
    assert stage_of_op("Matmul", prof) == "prompt", "the bare code still ranks by cost"


def test_the_stop_gate_is_the_registered_tool_and_the_helper_is_not():
    """The helper sits directly above termination_check, where a stray decorator unregisters it.

    termination_check is the binding stop gate; an agent that cannot call it cannot finish a round.
    stage_of_op is a shared helper and has no business on the wire.
    """
    src = (PERF / "cc_optimize" / "perf_mcp.py").read_text(encoding="utf-8")
    assert "@mcp.tool()\ndef termination_check(" in src, "the stop gate must stay registered"
    assert "@mcp.tool()\ndef stage_of_op(" not in src, "a shared helper must not be exposed as a tool"


# ---------------------------------------------------------------- glue ops: attributed by neighbour


def _prof_with_glue(stages: dict, glue_op: str, prev_op: str = "", next_op: str = ""):
    """A capture where `glue_op` sits in the real op stream (`buckets`, tagged with its neighbours)
    but was never dispatched by the per-stage replay, so it has nothing of its own in stage_buckets --
    the on-device argmax case: it runs in the generation loop, not inside any stage's own trace-step."""
    return {
        "stage_buckets": {
            name: [{"top_ops": [{"op_code": c, "device_ms": ms} for c, ms in ops]}] for name, ops in stages.items()
        },
        "buckets": [{"top_ops": [{"op_code": glue_op, "prev_op": prev_op, "next_op": next_op, "device_ms": 0.3}]}],
    }


def test_a_glue_op_is_attributed_to_its_neighbours_stage():
    """The exact nemotron case: ArgMax never appears in stage_buckets at all -- it runs in the
    generation loop, after decode_trace_step returns -- but the real capture tags it with what ran
    right before it, and that op IS attributed."""
    prof = _prof_with_glue({"decode": [("Matmul", 100.0)]}, "ArgMax", prev_op="Matmul")
    assert stage_of_op("ArgMax", prof) == "decode"


def test_a_direct_match_is_never_overridden_by_a_neighbour():
    """An op with its own evidence must not be re-attributed by what happens to sit next to it in
    the real capture -- the neighbour fallback is for ops with NO evidence of their own, not a
    second vote on ops that already have one."""
    prof = _prof_with_glue({"decode": [("X", 5.0)], "prefill": [("X", 40.0)]}, "X", prev_op="Y")
    prof["buckets"][0]["top_ops"].append({"op_code": "Y", "device_ms": 1.0})
    prof["stage_buckets"]["decode"][0]["top_ops"].append({"op_code": "Y", "device_ms": 90.0})
    assert stage_of_op("X", prof) == "prefill", "X's own ranking wins; Y's stage must not leak in"


def test_prev_op_is_tried_before_next_op():
    prof = _prof_with_glue({"before": [("P", 1.0)], "after": [("N", 1.0)]}, "ArgMax", prev_op="P", next_op="N")
    assert stage_of_op("ArgMax", prof) == "before"


def test_next_op_is_used_when_prev_op_has_no_evidence():
    prof = _prof_with_glue({"after": [("N", 1.0)]}, "ArgMax", prev_op="Nowhere", next_op="N")
    assert stage_of_op("ArgMax", prof) == "after"


def test_a_glue_op_whose_neighbours_are_also_unattributable_says_nothing():
    """Present in the real capture, but neither neighbour resolves to a stage -- honest silence,
    not a guess."""
    prof = _prof_with_glue({}, "ArgMax", prev_op="Nowhere", next_op="AlsoNowhere")
    assert stage_of_op("ArgMax", prof) == ""


def test_a_neighbour_tie_says_nothing_too():
    """The neighbour fallback answers through the SAME rule direct matches do -- a tie is a tie
    wherever it is found."""
    prof = _prof_with_glue({"one": [("P", 5.0)], "two": [("P", 5.0)]}, "ArgMax", prev_op="P")
    assert stage_of_op("ArgMax", prof) == ""


def test_an_op_absent_from_the_real_capture_too_says_nothing():
    """No stage_buckets entry AND no buckets entry -- there is nothing here to fall back to, so this
    must behave exactly as it did before the fallback existed."""
    prof = {"stage_buckets": {"decode": [{"top_ops": [{"op_code": "Matmul", "device_ms": 1.0}]}]}, "buckets": []}
    assert stage_of_op("NeverCaptured", prof) == ""
