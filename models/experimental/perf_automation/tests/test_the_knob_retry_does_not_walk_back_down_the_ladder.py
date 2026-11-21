"""A second knob variant is for an op still ON the knob rungs, not one that already went deeper.

The ladder is knob -> structural -> tt-lang -> C++, and _MAX_KNOB_RETRIES=2 lets a PREFERRED knob get
a second variant: the first grid attempt reads the profile, the second acts on what it learned ("2nd
grid variant: lifted the core-budget cap"). That is worth having.

But the allowance is counted per-knob with no reference to how far the op has already climbed:

    if tries[knob] >= (_MAX_KNOB_RETRIES if want_preferred else 1):

so an op that has ALREADY had a C++ kernel written for it is still owed a second grid try. On
gemma-3-12b-it, NLPConcatHeads carried grid=374.56 and cpp=366.85 from earlier runs; run 22 resumed,
saw grid_tries==1 < 2, and handed out grid a third time. It measured 377.08 -- slower than both, on
an op whose deepest rung was already spent.

Going back down is not a cheap re-check. It costs a full round (edit + PCC + end-to-end measurement)
on an op the ladder has already finished with, while genuinely untried ops wait.

So: a deeper rung on file spends the knob allowance. Once structural/tt-lang/cpp/tp-fracture has a
clean attempt, each knob is offered once and no more. An op still on the knob rungs is untouched --
the second variant is exactly where it earns its keep.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _attempt(kind, sig="NLPConcatHeadsDeviceOperation", ms=374.56):
    return {"op_signature": sig, "kernel_kind": kind, "measured_ms": ms}


def _open_op(**kw):
    op = {"op": "NLPConcatHeadsDeviceOperation", "grid": "partial", "bound_by": "compute", "gap_ms": 8.0}
    op.update(kw)
    return op


def _rung(mcp, attempts, open_op=None, op_code="NLPConcatHeadsDeviceOperation"):
    done, rung, _reason = mcp._op_ladder_status(open_op or _open_op(), op_code, attempts)
    return done, rung


# ---------------------------------------------------------------- the reported case


def test_grid_is_not_offered_a_third_time_after_cpp(mcp):
    """NLPConcatHeads: grid + cpp on file from earlier runs, and run 22 was handed grid again."""
    done, rung = _rung(mcp, [_attempt("grid"), _attempt("cpp", ms=366.85)])
    assert rung != "knob:grid", rung


def test_whatever_is_offered_next_has_never_been_tried(mcp):
    """The rule is about REPEATS, not about closing the knob rungs. With grid+cpp on file the ladder
    may still offer fidelity/dtype/shard -- none of them has been tried -- but never a rung that
    already has an attempt against it."""
    tried = {"grid", "cpp"}
    done, rung = _rung(mcp, [_attempt(k) for k in tried])
    assert str(rung).replace("knob:", "") not in tried, rung


@pytest.mark.parametrize("deep", ["structural", "tt-lang", "cpp", "tp-fracture"])
def test_any_deeper_rung_spends_the_allowance(mcp, deep):
    done, rung = _rung(mcp, [_attempt("grid"), _attempt(deep)])
    assert rung != "knob:grid", (deep, rung)


# ---------------------------------------------------------------- the second variant still works


def test_a_second_grid_variant_is_still_offered_on_the_knob_rungs(mcp):
    """The case the allowance exists for: one grid attempt, nothing deeper. Matmul 1024x3840x8192's
    '2nd grid variant: lifted the core-budget cap' must still be reachable."""
    done, rung = _rung(mcp, [_attempt("grid")])
    assert rung == "knob:grid", rung


def test_a_first_attempt_is_always_offered(mcp):
    done, rung = _rung(mcp, [])
    assert rung == "knob:grid", rung


def test_a_knob_never_tried_is_still_offered_after_a_deep_rung(mcp):
    """Spending the RETRY allowance is not the same as sealing the rung. A knob with zero attempts
    has not been tried at all, and the ladder still owes it one."""
    done, rung = _rung(mcp, [_attempt("grid"), _attempt("cpp")], open_op=_open_op(grid="full", bound_by="compute"))
    assert rung == "knob:fidelity", rung


# ---------------------------------------------------------------- nothing else moves


def test_the_deeper_rungs_are_unaffected(mcp):
    """With the knobs done and no structural attempt, structural is still what comes next."""
    done, rung = _rung(mcp, [_attempt("grid"), _attempt("fidelity"), _attempt("shard")], _open_op(grid="full"))
    assert rung in ("structural", "knob:fidelity", "knob:shard"), rung


def test_an_exhausted_op_still_finishes(mcp):
    every = [_attempt(k) for k in ("grid", "fidelity", "shard", "dtype", "structural", "tt-lang", "cpp")]
    done, _rung_name = _rung(mcp, every, _open_op(grid="full"))
    assert done is True
