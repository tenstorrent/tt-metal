"""A diverged reading never becomes a verdict, but it must still close the rung it was tried on.

record_kernel_attempt correctly refuses to bank a "diverged" full-pipeline reading (>8% slower than
the best-ever, PERF_MCP_FULLPIPE_TOL) as a measured win or loss: the comment at
_attempt_fullpipe_verdict says why -- in this harness that is nearly always the degraded-device
regime, not the edit. But refusing the RECORD too meant the attempt left no trace at all, so
_rung_allowance (which can only count rows that exist) never saw it. On nemotron's lm_head matmul:
284 of 305 check_full_pipeline_latency calls came back diverged, every record_kernel_attempt for them
was refused, and dtype/block/shard still read 0 tries after ~14 hours -- the same op offered forever,
the same shape of bug fix #5 already solved for a wedge (a crash also has no measured_ms, and used to
be invisible to the same counter for the same reason).

The fix mirrors fix #5 exactly: _autorecord_diverged gives the attempt a row shaped like
_autorecord_wedge's (measured_ms stays None, nothing is banked), and _rung_allowance /
_op_ladder_status count it -- but only when the reading was not itself produced by a clamped board
(_measure_full_pipeline_guarded already discards and retries a clamped reading, so this can only
matter if PERF_MCP_THERMAL_GATE=0 lets one through).
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
    monkeypatch.delenv("PERF_MCP_ALLOW_UNMEASURED_ATTEMPT", raising=False)
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


_SEQ = {"n": 0}


def _history(mcp, rows):
    Path(mcp._KERNEL_LOG_PATH).write_text(json.dumps(rows))
    Path(str(mcp._KERNEL_LOG_PATH) + ".cumulative").write_text(json.dumps(rows))


def _diverged(sig, kind, clamped=False):
    """A diverged record, shaped like _autorecord_diverged actually writes one: no measured_ms (a
    diverged reading is never banked), wedged always False (it is not a crash)."""
    _SEQ["n"] += 1
    return {
        "op_signature": sig,
        "kernel_kind": kind,
        "measured_ms": None,
        "beat_baseline": False,
        "wedged": False,
        "diverged": True,
        "board_clamped": clamped,
        "kernel_detected_in_source": False,
        "note": "diverged %d" % _SEQ["n"],
    }


# ---------------------------------------------------------------- _rung_allowance counts a diverged read


def test_a_diverged_read_counts_the_same_as_a_measured_try(mcp):
    _history(mcp, [_diverged("Matmul A", "dtype")])
    tries, allowed = mcp._rung_allowance("Matmul A", "dtype", mcp._load_attempts_all())
    assert (tries, allowed) == (1, mcp._MAX_KNOB_RETRIES)


def test_a_diverged_read_closes_a_knob_rung_once_the_cap_is_spent(mcp):
    _history(mcp, [_diverged("Matmul A", "dtype"), _diverged("Matmul A", "dtype")])
    tries, allowed = mcp._rung_allowance("Matmul A", "dtype", mcp._load_attempts_all())
    assert tries >= allowed


def test_a_diverged_read_closes_a_deep_rung_immediately(mcp):
    _history(mcp, [_diverged("Matmul A", "cpp")])
    tries, allowed = mcp._rung_allowance("Matmul A", "cpp", mcp._load_attempts_all())
    assert (tries, allowed) == (1, 1)


def test_a_clamped_board_does_not_close_the_rung(mcp):
    """PERF_MCP_THERMAL_GATE=0 is the only way a clamped reading reaches here at all; even then, it
    must not spend the rung's real attempts -- a hot board making a reading look bad is not the lever
    being genuinely tried."""
    _history(mcp, [_diverged("Matmul A", "dtype", clamped=True), _diverged("Matmul A", "dtype", clamped=True)])
    tries, allowed = mcp._rung_allowance("Matmul A", "dtype", mcp._load_attempts_all())
    assert tries < allowed


def test_a_diverged_read_on_a_different_rung_does_not_close_this_one(mcp):
    _history(mcp, [_diverged("Matmul A", "block")])
    tries, _allowed = mcp._rung_allowance("Matmul A", "dtype", mcp._load_attempts_all())
    assert tries == 0


def test_a_diverged_read_on_a_different_op_is_untouched(mcp):
    _history(mcp, [_diverged("Matmul A", "dtype")])
    tries, _allowed = mcp._rung_allowance("LayerNorm", "dtype", mcp._load_attempts_all())
    assert tries == 0


# ---------------------------------------------------------------- _op_ladder_status sees it too
#
# At the level fix #5 (the wedge precedent) itself tested: _rung_allowance directly, not through
# _op_ladder_status's bound-conditional knob ordering, which decides WHICH rung is tried first for
# reasons unrelated to this fix (fidelity vs. shard for a compute- vs memory-bound op) and would make
# an integration-level assertion fragile against logic this change never touches.


def test_op_ladder_status_counts_a_diverged_read_from_the_full_log(mcp):
    """The same blind spot _op_ladder_status already had for a wedge (perf_mcp.py:1594) -- read from
    _load_attempts_all, not the caller's kernel_detected_in_source-filtered `attempts`, because a
    diverged reading never populates that field either."""
    _history(mcp, [_diverged("Matmul A", "cpp"), _diverged("Matmul A", "cpp")])
    open_op = {"op_code": "Matmul A", "grid": "full", "weight_dtype": "bf16", "fidelity": "hifi4", "bound_by": "memory"}
    # attempts=[] mirrors the wedge case exactly: kernel_detected_in_source=False means the caller's
    # own filtered list never contains it, so only the full-log read (which this test exercises) can
    # see it at all.
    _done, rung, _reason = mcp._op_ladder_status(open_op, "Matmul A", [])
    assert mcp._normalise_rung(rung) != "cpp", "cpp already had its one permitted attempt: got %r" % rung


# ---------------------------------------------------------------- record_kernel_attempt writes the row


def test_a_diverged_verdict_is_recorded_as_diverged_not_a_verdict(mcp, monkeypatch):
    monkeypatch.setattr(mcp, "_attempt_fullpipe_verdict", lambda: {"own": False, "ms": None, "ref": None})
    monkeypatch.setattr(
        mcp,
        "gate_verdicts",
        lambda: {"full_pipeline": {"status": "diverged", "full_pipeline_ms": 516.0, "measurement_id": "fp-1"}},
    )
    out = mcp.record_kernel_attempt("Matmul A", "dtype", 350.0, False, note="tried bf8_b")
    assert out.get("recorded") is False and "owns no end-to-end measurement" in (out.get("refused") or "")
    rows = mcp._load_attempts_all()
    assert len(rows) == 1
    row = rows[0]
    assert row["diverged"] is True
    assert row["measured_ms"] is None, "a diverged reading must never be banked as a real number"
    assert row.get("beat_baseline") is not True


def test_the_same_unmeasured_verdict_does_not_mint_a_second_row(mcp, monkeypatch):
    """Two record_kernel_attempt calls against the SAME diverged reading (no fresh
    check_full_pipeline_latency in between) must be one attempt, not two."""
    monkeypatch.setattr(mcp, "_attempt_fullpipe_verdict", lambda: {"own": False, "ms": None, "ref": None})
    monkeypatch.setattr(
        mcp,
        "gate_verdicts",
        lambda: {"full_pipeline": {"status": "diverged", "full_pipeline_ms": 516.0, "measurement_id": "fp-1"}},
    )
    mcp.record_kernel_attempt("Matmul A", "dtype", 350.0, False, note="try 1")
    mcp.record_kernel_attempt("Matmul A", "dtype", 350.0, False, note="try 2")
    assert len(mcp._load_attempts_all()) == 1


def test_a_fresh_diverged_measurement_does_mint_a_new_row(mcp, monkeypatch):
    """A genuinely new measurement (a new measurement_id) is a real second attempt."""
    monkeypatch.setattr(mcp, "_attempt_fullpipe_verdict", lambda: {"own": False, "ms": None, "ref": None})
    calls = {"n": 0}

    def _gv():
        calls["n"] += 1
        return {
            "full_pipeline": {"status": "diverged", "full_pipeline_ms": 516.0, "measurement_id": "fp-%d" % calls["n"]}
        }

    monkeypatch.setattr(mcp, "gate_verdicts", _gv)
    mcp.record_kernel_attempt("Matmul A", "dtype", 350.0, False, note="try 1")
    mcp.record_kernel_attempt("Matmul A", "dtype", 350.0, False, note="try 2")
    assert len(mcp._load_attempts_all()) == 2


def test_a_regressed_verdict_is_not_treated_as_diverged(mcp, monkeypatch):
    """The existing 'regressed' path (a real, ownable loss) must keep working exactly as before --
    this fix only adds a row for status=='diverged', never touches 'ok'/'regressed'."""
    monkeypatch.setattr(
        mcp,
        "_attempt_fullpipe_verdict",
        lambda: {"own": True, "ms": 420.0, "ref": 400.0, "delta": 20.0, "win": False, "metric": "device_ms"},
    )
    monkeypatch.setattr(mcp, "gate_verdicts", lambda: {"full_pipeline": {"status": "regressed"}})
    out = mcp.record_kernel_attempt("Matmul A", "dtype", 420.0, False, note="slower but measured")
    assert out.get("recorded") is not False
    rows = mcp._load_attempts_all()
    assert len(rows) == 1 and not rows[0].get("diverged")


# ---------------------------------------------------------------- the core safety property


def test_a_diverged_row_never_counts_as_a_win():
    """The property the whole design hinges on: however it is tallied downstream, a diverged row
    cannot be a new best -- it stamps no fullpipe_delta_ms and beat_baseline stays False."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cc_optimize"))
    from measurements import winning_indices

    real_win = {"measured_ms": 100.0, "beat_baseline": True, "fullpipe_delta_ms": -50.0}
    row = _diverged("Matmul A", "dtype")
    wins = winning_indices([real_win, row], baseline_ms=150.0)
    assert wins == {0}, "only the real, stamped win may appear in the winning set"


def test_a_diverged_row_never_counts_as_a_loss_either():
    """No fullpipe_delta_ms means winning_indices' stamped-verdict path cannot see it as a positive
    delta (a loss) any more than as a negative one (a win) -- it is simply absent from the reckoning."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cc_optimize"))
    from measurements import winning_indices

    row = _diverged("Matmul A", "dtype")
    another = _diverged("Matmul A", "dtype")
    wins = winning_indices([row, another], baseline_ms=150.0)
    assert wins == set()
