"""A knob that failed accuracy on the whole repeated stack is offered again on PART of it.

A repeated block is one shared definition, so a fidelity or dtype edit on it lowers every instance at
once. An image-edit port on a 32-chip mesh (2026-10-13) recorded 19 such tries, every one reverted at
PCC 0.9443-0.9470 against a 0.95 bar -- and none on fewer layers, because nothing offered it: the
ladder closed the knob on the whole-stack result and coverage called any subset a defect.

These pin the subset walk: owed only after a whole-stack pcc_low, only on an op with instances,
bounded, its own allowance, its history handed back, and coverage judged against the intended subset.
Nothing here touches a device.
"""

import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

_OP = "MatmulDeviceOperation 512 x 3072 x 1536"


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_KERNEL_LOG", str(tmp_path / "kl.json"))
    monkeypatch.delenv("PERF_MCP_ALLOW_RETRIED_RUNG", raising=False)
    monkeypatch.delenv("PERF_MCP_MAX_PARTIAL_TRIES", raising=False)
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    monkeypatch.setattr(m, "_attempt_fullpipe_verdict", lambda: {"own": True, "ms": 35.0, "ref": 35.0, "delta": 0.0})
    monkeypatch.setattr(m, "_attempt_pcc_verdict", lambda: {})
    return m


_SEQ = {"n": 0}


def _row(kind, **kw):
    _SEQ["n"] += 1
    r = {
        "op_signature": _OP,
        "kernel_kind": kind,
        "measured_ms": 1200.0 + _SEQ["n"],
        "note": "attempt %d" % _SEQ["n"],
        "kernel_detected_in_source": True,
    }
    r.update(kw)
    return r


def _pcc_low(kind="fidelity", **kw):
    return _row(kind, pcc_status="pcc_low", pcc=0.9462, pcc_threshold=0.95, **kw)


def _history(mcp, rows):
    Path(mcp._KERNEL_LOG_PATH).write_text(json.dumps(rows))
    Path(str(mcp._KERNEL_LOG_PATH) + ".cumulative").write_text(json.dumps(rows))


def _op(count=120):
    # past every knob: full grid, one tile row (no block rung), already gone deep (one try per knob)
    return {"grid": "full", "bound_by": "compute", "weight_dtype": "bf16", "count": count}


def _past_knobs(fidelity_row):
    """An op that has climbed past the knobs: each knob tried once and a deep rung on file."""
    return [_row("cpp"), _row("grid"), _row("dtype"), _row("shard"), fidelity_row]


def _ladder(mcp, rows, count=120):
    _history(mcp, rows)
    return mcp._op_ladder_status(_op(count), _OP, mcp._load_attempts_all())


# ---------------------------------------------------------------- when the walk is owed


def test_a_whole_stack_pcc_failure_reopens_the_knob_on_a_subset(mcp):
    done, rung, reason = _ladder(mcp, _past_knobs(_pcc_low()))
    assert (done, rung) == (False, "knob:fidelity")
    assert reason.startswith(mcp._PARTIAL_LEAD), reason


def test_a_knob_that_was_only_slower_is_not_reopened(mcp):
    """No accuracy failure says nothing about which instances carry the error."""
    _done, rung, reason = _ladder(mcp, _past_knobs(_row("fidelity")))
    assert not reason.startswith(mcp._PARTIAL_LEAD) and rung != "knob:fidelity"


def test_an_op_seen_once_has_no_subset_to_take(mcp):
    _done, _rung, reason = _ladder(mcp, _past_knobs(_pcc_low()), count=1)
    assert not reason.startswith(mcp._PARTIAL_LEAD)


def test_the_dtype_knob_walks_the_same_way(mcp):
    rows = [_row("cpp"), _row("grid"), _row("fidelity"), _row("shard"), _pcc_low("dtype")]
    _done, rung, reason = _ladder(mcp, rows)
    assert rung == "knob:dtype" and reason.startswith(mcp._PARTIAL_LEAD)


def test_a_grid_or_shard_pcc_failure_is_not_a_precision_budget(mcp):
    """Those knobs do not change the arithmetic; failing PCC there means the edit was wrong."""
    for kind in ("grid", "shard"):
        rows = [_row("cpp")] + [_row(k) for k in ("grid", "fidelity", "dtype", "shard") if k != kind] + [_pcc_low(kind)]
        _done, _rung, reason = _ladder(mcp, rows)
        assert not reason.startswith(mcp._PARTIAL_LEAD), kind


# ---------------------------------------------------------------- sized by the headroom


def _ok(pcc):
    return _row("cpp", op_signature="SomeOtherOp", pcc_status="ok", pcc=pcc)


def test_the_first_subset_is_sized_by_headroom_over_loss(mcp):
    # headroom 0.9520 - 0.95 = 0.002, whole-stack loss 0.9520 - 0.9462 = 0.0058 -> ~34.5%
    _done, _rung, reason = _ladder(mcp, [_ok(0.9520)] + _past_knobs(_pcc_low()))
    assert reason.startswith(mcp._PARTIAL_LEAD) and "about 34.5%" in reason, reason


def test_no_headroom_means_no_subset_can_pass(mcp):
    _done, _rung, reason = _ladder(mcp, [_ok(0.95)] + _past_knobs(_pcc_low()))
    assert not reason.startswith(mcp._PARTIAL_LEAD)


def test_a_miss_too_big_for_even_one_instance_is_not_walked(mcp):
    """The measured case: 0.9501 against 0.95 and a whole-stack 0.9462 fits ~2.6% of the instances --
    not one of 16, but 3 of 120."""
    rows = [_ok(0.9501)] + _past_knobs(_pcc_low())
    assert not _ladder(mcp, rows, count=16)[2].startswith(mcp._PARTIAL_LEAD)
    assert _ladder(mcp, rows, count=120)[2].startswith(mcp._PARTIAL_LEAD)


def test_without_a_passing_reading_the_walk_starts_at_half(mcp):
    _done, _rung, reason = _ladder(mcp, _past_knobs(_pcc_low()))
    assert "lower half of the instances" in reason


# ---------------------------------------------------------------- bounded, and it remembers


def test_the_walk_stops_after_its_allowance(mcp):
    rows = _past_knobs(_pcc_low()) + [
        _row("fidelity", applied_blocks=list(range(i, i + 10))) for i in range(mcp._MAX_PARTIAL_TRIES)
    ]
    _done, _rung, reason = _ladder(mcp, rows)
    assert not reason.startswith(mcp._PARTIAL_LEAD)


def test_a_wedged_or_diverged_subset_try_still_counts(mcp):
    """Absent from the caller's filtered attempts, so read from the full log -- else a subset that
    hangs every time is offered forever."""
    rows = _past_knobs(_pcc_low()) + [
        _row("fidelity", measured_ms=None, wedged=True, kernel_detected_in_source=False, applied_blocks="unstated")
        for _ in range(mcp._MAX_PARTIAL_TRIES)
    ]
    _done, _rung, reason = _ladder(mcp, rows)
    assert not reason.startswith(mcp._PARTIAL_LEAD)


def test_each_try_is_handed_the_earlier_subsets_and_their_pcc(mcp):
    rows = _past_knobs(_pcc_low()) + [
        _row("fidelity", applied_blocks=[0, 1, 2], pcc_status="pcc_low", pcc=0.9481),
    ]
    _done, _rung, reason = _ladder(mcp, rows)
    assert "[0, 1, 2]" in reason and "0.9481" in reason and "(2 of %d)" % mcp._MAX_PARTIAL_TRIES in reason


def test_subset_tries_do_not_spend_the_whole_stack_retries(mcp):
    rows = [_row("grid", applied_blocks=[0])] * 5
    _history(mcp, rows)
    tries, allowed = mcp._rung_allowance(_OP, "grid", mcp._load_attempts_all())
    assert tries == 0 and allowed == mcp._MAX_KNOB_RETRIES


# ---------------------------------------------------------------- the record carries the subset


def test_a_subset_try_is_recorded_with_its_blocks(mcp):
    _history(mcp, _past_knobs(_pcc_low()))
    out = mcp.record_kernel_attempt(_OP, "fidelity", 1199.0, False, note="half", applied_blocks="0-29")
    assert out.get("recorded") is True, out
    assert out["attempt"]["applied_blocks"] == list(range(30))


def test_a_subset_try_with_no_pcc_failure_on_file_is_refused(mcp):
    _history(mcp, _past_knobs(_row("fidelity")))
    out = mcp.record_kernel_attempt(_OP, "fidelity", 1199.0, False, applied_blocks="0-29")
    assert out.get("recorded") is False and "subset" in out["refused"], out


def test_a_subset_on_a_kernel_rung_is_refused(mcp):
    _history(mcp, [_pcc_low("cpp")])
    out = mcp.record_kernel_attempt(_OP, "cpp", 1199.0, False, applied_blocks="0-3")
    assert out.get("recorded") is False, out


def test_unreadable_blocks_are_refused(mcp):
    _history(mcp, _past_knobs(_pcc_low()))
    out = mcp.record_kernel_attempt(_OP, "fidelity", 1199.0, False, applied_blocks="first half")
    assert out.get("recorded") is False and "applied_blocks" in out["refused"]


def test_a_whole_stack_record_is_unchanged(mcp):
    _history(mcp, [])
    out = mcp.record_kernel_attempt(_OP, "fidelity", 1199.0, False, note="whole")
    assert out.get("recorded") is True and "applied_blocks" not in out["attempt"]


@pytest.mark.parametrize(
    "spec, want",
    [("", []), ("0-2,5", [0, 1, 2, 5]), ("[3, 1, 1]", [1, 3]), ([4, 2], [2, 4]), ("x", None), ("-1", None)],
)
def test_block_specs(mcp, spec, want):
    assert mcp._parse_blocks(spec) == want


# ---------------------------------------------------------------- a try that never got to state its blocks


def test_a_wedge_from_a_subset_target_is_marked_a_subset_try(mcp):
    _history(mcp, [])
    mcp._persist_target({"op": _OP, "rung": "knob:fidelity", "reason": mcp._PARTIAL_LEAD + " (1 of 3): ..."})
    mcp._autorecord_wedge("TT_FATAL: hang")
    assert mcp._load_attempts()[-1].get("applied_blocks") == mcp._BLOCKS_UNSTATED


def test_a_wedge_from_any_other_target_is_not(mcp):
    _history(mcp, [])
    mcp._persist_target({"op": _OP, "rung": "knob:fidelity", "reason": "lower the math fidelity"})
    mcp._autorecord_wedge("TT_FATAL: hang")
    assert "applied_blocks" not in mcp._load_attempts()[-1]


# ---------------------------------------------------------------- coverage judges the subset it meant


def _signposted(dtypes):
    seq = ["embed(...)"]
    for i, dt in enumerate(dtypes):
        seq += ["PERF_BLOCK_SIGNPOST:%d" % i, "norm(...)", "matmul(((1,4096),),'%s')" % dt]
    return seq


def test_coverage_says_as_intended_for_exactly_the_intended_blocks(mcp):
    seq = _signposted(["BFLOAT8_B", "BFLOAT8_B", "BFLOAT16", "BFLOAT16"])
    r = mcp.compute_lever_coverage(Counter(seq), seq, "matmul(", "BFLOAT16", "BFLOAT8_B", [0, 1])
    assert r["as_intended"] is True and r["applied_blocks"] == [0, 1]
    assert r["fully_applied"] is False  # unchanged meaning: the subset is not the whole stack


def test_coverage_catches_a_subset_that_landed_elsewhere(mcp):
    seq = _signposted(["BFLOAT8_B", "BFLOAT16", "BFLOAT8_B", "BFLOAT16"])
    r = mcp.compute_lever_coverage(Counter(seq), seq, "matmul(", "BFLOAT16", "BFLOAT8_B", [0, 1])
    assert r["as_intended"] is False and r["applied_blocks"] == [0, 2]


def test_coverage_without_a_subset_is_unchanged(mcp):
    seq = _signposted(["BFLOAT8_B"] * 3)
    r = mcp.compute_lever_coverage(Counter(seq), seq, "matmul(", "BFLOAT16", "BFLOAT8_B")
    assert r["fully_applied"] is True and "as_intended" not in r


# ---------------------------------------------------------------- the agent is told about it


def test_the_agent_prompt_names_the_subset_exception():
    src = (Path(__file__).resolve().parents[1] / "cc_optimize" / "run.py").read_text()
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    assert src.count(m._PARTIAL_LEAD) >= 2 and "as_intended" in src
