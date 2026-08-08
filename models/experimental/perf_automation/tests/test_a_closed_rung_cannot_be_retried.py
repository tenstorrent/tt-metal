"""A rung that has had its attempts does not get another one. Enforced, not advised.

termination_check computes a DETERMINISTIC next_target and an exhausted op is moved out of
`blocking` into `cleared`, so the gate's own bookkeeping is sound. But what it returns is ADVICE,
and the agent is an LLM that routinely works on something else:

  - matmul was excluded twice by seeding conclusive attempts; runs 25 and 26 measured matmul anyway
  - NLPConcatHeadsDeviceOperation/shard already had `cpp` on file (deep rung -> knob cap 1) and
    three prior attempts; run 27 measured it a fourth time

Across gemma-3-12b-it's recorded history that is 30 repeats in 146 attempts (21%), and at its worst
62 in 162 (38%):

    MatmulDeviceOperation 128 x 15360 x 3840 / grid    x3
    MatmulDeviceOperation 32 x 15360 x 3840  / shard   x3
    BinaryNgDeviceOperation                  / grid    x3

Seeding the ladder cannot fix this -- it is what was tried, and it did not work.

record_kernel_attempt is the one choke point every attempt must pass through, so the refusal lives
there. A refused attempt cannot enter history, cannot clear an op, and cannot be banked as a win, so
re-doing a closed rung stops being discouraged and becomes unrecordable. The refusal names the
remaining rungs, so the agent is redirected rather than merely blocked -- the same shape as the
existing "this attempt owns no end-to-end measurement" refusal.

Permanence across RUNS comes from _load_attempts_all (archive UNION live): a rung tried in a previous
optimize of the same model is still closed in the next one. Permanence across ITERATIONS within a run
comes from the same read, since each attempt is appended as it happens.

The allowance mirrors _op_ladder_status rather than inventing new policy: knob rungs get
_MAX_KNOB_RETRIES, deep rungs get one, and a knob is capped to one once the op has gone deep. Failed
measurements do not count -- nothing was learned.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_KERNEL_LOG", str(tmp_path / "kl.json"))
    monkeypatch.delenv("PERF_MCP_ALLOW_RETRIED_RUNG", raising=False)
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


_SEQ = {"n": 0}


def _row(sig, kind, **kw):
    """Distinct rows. _load_attempts_all dedupes on (op_signature, kernel_kind, measured_ms, note),
    because _fold_cumulative copies live rows into the archive and the overlap must not double-count
    -- so two byte-identical rows ARE one attempt. Real attempts differ in what they measured."""
    _SEQ["n"] += 1
    r = {
        "op_signature": sig,
        "kernel_kind": kind,
        "measured_ms": 400.0 + _SEQ["n"],
        "note": "attempt %d" % _SEQ["n"],
        "kernel_detected_in_source": True,
    }
    r.update(kw)
    return r


def _history(mcp, rows):
    Path(mcp._KERNEL_LOG_PATH).write_text(json.dumps(rows))
    Path(str(mcp._KERNEL_LOG_PATH) + ".cumulative").write_text(json.dumps(rows))


# ---------------------------------------------------------------- the allowance mirrors the ladder


def test_a_knob_gets_its_permitted_retries(mcp):
    """_MAX_KNOB_RETRIES is 2, so one prior attempt still leaves one."""
    _history(mcp, [_row("Matmul A", "grid")])
    tries, allowed = mcp._rung_allowance("Matmul A", "grid", mcp._load_attempts_all())
    assert (tries, allowed) == (1, mcp._MAX_KNOB_RETRIES)


def test_a_knob_closes_once_its_retries_are_spent(mcp):
    _history(mcp, [_row("Matmul A", "grid"), _row("Matmul A", "grid")])
    tries, allowed = mcp._rung_allowance("Matmul A", "grid", mcp._load_attempts_all())
    assert tries >= allowed


def test_a_knob_is_capped_to_one_once_the_op_went_deep(mcp):
    """The NLPConcatHeads case: cpp on file means the knob search is over."""
    _history(mcp, [_row("ConcatHeads", "grid"), _row("ConcatHeads", "cpp")])
    tries, allowed = mcp._rung_allowance("ConcatHeads", "grid", mcp._load_attempts_all())
    assert allowed == 1 and tries >= allowed


def test_a_deep_rung_gets_exactly_one_attempt(mcp):
    _history(mcp, [_row("Matmul A", "cpp")])
    tries, allowed = mcp._rung_allowance("Matmul A", "cpp", mcp._load_attempts_all())
    assert (tries, allowed) == (1, 1)


def test_a_failed_measurement_does_not_burn_the_allowance(mcp):
    """Nothing was learned, so it must not close the rung."""
    _history(mcp, [_row("Matmul A", "cpp", measurement_failed=True)])
    attempts = [a for a in mcp._load_attempts_all() if not a.get("measurement_failed")]
    tries, allowed = mcp._rung_allowance("Matmul A", "cpp", attempts)
    assert tries < allowed


def test_a_different_rung_on_the_same_op_is_untouched(mcp):
    _history(mcp, [_row("Matmul A", "grid"), _row("Matmul A", "grid")])
    tries, _allowed = mcp._rung_allowance("Matmul A", "fidelity", mcp._load_attempts_all())
    assert tries == 0


def test_a_different_op_at_the_same_rung_is_untouched(mcp):
    _history(mcp, [_row("Matmul A", "grid"), _row("Matmul A", "grid")])
    tries, _allowed = mcp._rung_allowance("LayerNorm", "grid", mcp._load_attempts_all())
    assert tries == 0


# ---------------------------------------------------------------- the refusal is what enforces it


def test_recording_a_closed_rung_is_refused(mcp, monkeypatch):
    """The 38%-repeat case: this is what makes re-doing a closed rung unrecordable."""
    _history(mcp, [_row("Matmul A", "grid"), _row("Matmul A", "grid")])
    monkeypatch.setattr(mcp, "_attempt_fullpipe_verdict", lambda: {"own": True, "ms": 35.0, "ref": 35.0, "delta": 0.0})
    out = mcp.record_kernel_attempt("Matmul A", "grid", 399.0, False, note="third go")
    assert out.get("recorded") is False and "CLOSED" in (out.get("refused") or "")


def test_the_refusal_names_what_is_still_open(mcp, monkeypatch):
    """Blocking without redirecting just makes the agent guess again."""
    _history(mcp, [_row("Matmul A", "grid"), _row("Matmul A", "grid")])
    monkeypatch.setattr(mcp, "_attempt_fullpipe_verdict", lambda: {"own": True, "ms": 35.0, "ref": 35.0, "delta": 0.0})
    out = mcp.record_kernel_attempt("Matmul A", "grid", 399.0, False)
    assert "cpp" in (out.get("rungs_still_open") or []), out


def test_a_refused_attempt_does_not_enter_history(mcp, monkeypatch):
    """If it were recorded anyway the ladder would still see a fourth attempt, and 'closed' would be
    a label rather than a rule."""
    _history(mcp, [_row("Matmul A", "grid"), _row("Matmul A", "grid")])
    monkeypatch.setattr(mcp, "_attempt_fullpipe_verdict", lambda: {"own": True, "ms": 35.0, "ref": 35.0, "delta": 0.0})
    before = len(mcp._load_attempts_all())
    mcp.record_kernel_attempt("Matmul A", "grid", 399.0, False)
    assert len(mcp._load_attempts_all()) == before


def test_an_open_rung_is_still_recorded(mcp, monkeypatch):
    """The gate must not block real work -- only repeats."""
    _history(mcp, [])
    monkeypatch.setattr(mcp, "_attempt_fullpipe_verdict", lambda: {"own": True, "ms": 35.0, "ref": 35.0, "delta": 0.0})
    out = mcp.record_kernel_attempt("Matmul A", "grid", 399.0, False, note="first")
    assert out.get("recorded") is not False, out


def test_the_escape_hatch_allows_a_deliberate_retry(mcp, monkeypatch):
    monkeypatch.setenv("PERF_MCP_ALLOW_RETRIED_RUNG", "1")
    _history(mcp, [_row("Matmul A", "grid"), _row("Matmul A", "grid")])
    monkeypatch.setattr(mcp, "_attempt_fullpipe_verdict", lambda: {"own": True, "ms": 35.0, "ref": 35.0, "delta": 0.0})
    out = mcp.record_kernel_attempt("Matmul A", "grid", 399.0, False)
    assert out.get("recorded") is not False


# ---------------------------------------------------------------- closed stays closed across runs


def test_a_rung_tried_in_a_PREVIOUS_run_is_still_closed(mcp, monkeypatch):
    """The operator's requirement. The live log is rewritten by the resume filter each run, so this
    only holds because the check reads archive UNION live."""
    live = []
    cum = [_row("Matmul A", "grid", baseline_at_record=381.22), _row("Matmul A", "grid", baseline_at_record=381.22)]
    Path(mcp._KERNEL_LOG_PATH).write_text(json.dumps(live))
    Path(str(mcp._KERNEL_LOG_PATH) + ".cumulative").write_text(json.dumps(cum))
    monkeypatch.setattr(mcp, "_attempt_fullpipe_verdict", lambda: {"own": True, "ms": 35.0, "ref": 35.0, "delta": 0.0})
    out = mcp.record_kernel_attempt("Matmul A", "grid", 399.0, False)
    assert out.get("recorded") is False, out


# ---------------------------------------------------------------- the ladder has ONE definition


def test_the_refusal_offers_the_host_rung(mcp):
    """`host` was missing from _LADDER_ORDER, so the refusal named every remaining rung EXCEPT the
    dispatch one -- on a model where every top op reads bound_by=dispatch and host_overhead is the
    second-largest bucket. Across 158 attempts the dispatch axis was tried once."""
    assert "host" in mcp.ladder_order(), mcp.ladder_order()


def test_the_ladder_has_a_single_definition(mcp):
    """summary.py used to restate the climb order as a hardcoded display string; writing this
    constant fresh from memory is exactly how the two drifted."""
    import importlib

    summary = importlib.import_module("models.experimental.perf_automation.cc_optimize.summary")
    rendered = summary._levels_display()
    for rung in mcp.ladder_order():
        if rung != "tt-lang":
            assert rung in rendered, "%s missing from the rendered levels: %s" % (rung, rendered)


def test_the_rendered_ladder_follows_the_profile_the_report_is_about(mcp):
    """The report printed ONE fixed order for every model. It now reads the binding off the same
    per-op `bound_by` annotation the ladder gate reads, weighted by gap_ms -- a hundred tiny eltwise
    ops must not outvote the matmul that owns the residual."""
    import importlib

    summary = importlib.import_module("models.experimental.perf_automation.cc_optimize.summary")
    prof = {
        "buckets": [
            {"top_ops": [{"bound_by": "memory", "gap_ms": 40.0}]},
            {"top_ops": [{"bound_by": "compute", "gap_ms": 1.0}] * 20},
        ]
    }
    assert summary._dominant_bound_by(prof) == "memory"
    assert summary._dominant_bound_by(None) == ""
    assert summary._dominant_bound_by({"buckets": [{"top_ops": [{"gap_ms": 5.0}]}]}) == ""
    rendered = summary._levels_display(summary._dominant_bound_by(prof))
    assert rendered.index("dtype") < rendered.index("fidelity"), rendered


def test_the_lever_that_can_move_the_number_leads(mcp):
    """THE CLIMB ORDER FOLLOWS THE BINDING, because a lever that cannot move the measurement is not
    a cheap first try -- it is a wasted round.

    fidelity speeds the MATH ENGINE. A fixed order put it second for every model, so on gemma-3-12b
    -- memory-bound in both stages, decode compute running at 0.1% of peak -- the ladder led with
    the one knob that cannot help, ahead of the two that cut bytes directly. And on a dispatch-bound
    op no knob applies at all: the op is waiting on the host loop that launches it."""
    assert mcp.ladder_order("memory").index("dtype") < mcp.ladder_order("memory").index("fidelity")
    assert mcp.ladder_order("compute").index("fidelity") < mcp.ladder_order("compute").index("dtype")
    assert mcp.ladder_order("dispatch")[0] == "host"


def test_the_binding_sets_priority_never_membership(mcp):
    """bound_by is a roofline ESTIMATE, ops are rarely purely one-bound, and `compute` is only ever
    computed for matmuls -- so no reduction/eltwise op can ever read compute-bound however it
    behaves. Used as a FILTER that silently deletes levers for a whole run: llama3_1_8b_p150
    recorded 0 fidelity attempts across 133, its two costliest ops structurally ineligible."""
    full = set(mcp.ladder_order())
    for bound in ("memory", "compute", "dispatch", "", "nonsense-bound"):
        assert set(mcp.ladder_order(bound)) == full, bound


def test_the_knob_order_is_not_a_second_ordering(mcp):
    """_KNOB_ORDER and the climb order were two literals over the same rungs and only one of them
    knew what the op was waiting on. Derived from one table now, so they cannot disagree."""
    for bound, knobs in mcp._KNOB_ORDER.items():
        assert list(knobs) == [r for r in mcp.ladder_order(bound) if r in mcp._KNOBS], bound


def test_the_climb_order_is_cheapest_first(mcp):
    """knobs before structural before hand-written kernels -- a long ladder must not spend its
    budget on tt-lang/C++ before reaching a cheaper restructure."""
    order = mcp.ladder_order()
    assert order.index("grid") < order.index("structural") < order.index("tt-lang") < order.index("cpp")
