"""One end-to-end measurement, one win. Ownership is a fact about who ran the trace, not a guess.

An e2e measurement costs a full trace replay (minutes), so it is deliberately NOT run per attempt --
attempts that did not trigger one must report own=False and carry no delta and no win.
_attempt_fullpipe_verdict exists to enforce exactly that, and it was defeated by its own key:

    def _verdict_identity(fp):
        mt = _gate_verdict_path().stat().st_mtime_ns     # rewritten on EVERY recorded verdict
        return [str(fp.get("sha") or ""), fp.get("full_pipeline_ms"), mt]

The mtime was meant to separate "a genuinely new measurement" from "the same one read again". But the
verdict file is rewritten whenever ANY gate records a verdict -- pcc, measure, commit -- not only when
a trace replay runs. So the mtime moves with no measurement behind it, the identity looks new, and the
next attempt claims a reading it never took. The sha component contributes nothing either: it is ""
in practice.

On gemma-3-12b-it that produced fourteen attempts, across five different ops, all carrying the
identical `35.2531 -> 35.0605, -0.1926` and all marked win -- one 0.19 ms improvement counted
fourteen times. One of the fourteen was a structural change that measured 2.1x SLOWER on device, was
PCC-rejected at 0.9315 and reverted by the agent, and still shows a win.

The fix is to key on the MEASUREMENT: the code that runs the trace replay stamps a fresh id, and only
a verdict carrying an unconsumed id can be owned. No id, no win.
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
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _verdict(m, ms, sha="", measurement_id=None, best_ms=None):
    """Record a full_pipeline verdict the way the gate does."""
    extra = {"full_pipeline_ms": ms, "sha": sha}
    if best_ms is not None:
        extra["best_ms"] = best_ms
    if measurement_id is not None:
        extra["measurement_id"] = measurement_id
    m.record_gate_verdict("full_pipeline", "ok", **extra)


def _touch_unrelated_verdict(m):
    """Any other gate recording anything -- this is what moved the mtime."""
    m.record_gate_verdict("pcc", "ok", pcc=0.99)


# ---------------------------------------------------------------- the reported bug


def test_a_second_attempt_cannot_claim_the_same_measurement(mcp):
    m = mcp
    _verdict(m, 35.0605, measurement_id="meas-1")
    first = m._attempt_fullpipe_verdict()
    second = m._attempt_fullpipe_verdict()
    assert first["own"] is True and first["ms"] == 35.0605
    assert second["own"] is False and second["win"] is False and second["delta"] is None


def test_an_unrelated_gate_verdict_does_not_re_open_ownership(mcp):
    """THE BUG. A pcc verdict rewrites the file, the mtime moves, and the next attempt claimed the
    e2e reading as its own. Fourteen times on gemma3."""
    m = mcp
    _verdict(m, 35.0605, measurement_id="meas-1")
    assert m._attempt_fullpipe_verdict()["own"] is True
    for _ in range(14):
        _touch_unrelated_verdict(m)
        again = m._attempt_fullpipe_verdict()
        assert again["own"] is False, "an unmeasured attempt claimed the reading again"
        assert again["win"] is False


def test_fourteen_attempts_yield_one_win(mcp):
    """The end-to-end shape of the reported failure, as a single assertion."""
    m = mcp
    _verdict(m, 35.0605, measurement_id="meas-1")
    wins = 0
    for _ in range(14):
        wins += 1 if m._attempt_fullpipe_verdict()["win"] else 0
        _touch_unrelated_verdict(m)
    assert wins <= 1, "%d attempts claimed the same measurement" % wins


# ---------------------------------------------------------------- a real new measurement still counts


def test_a_genuinely_new_measurement_is_owned(mcp):
    m = mcp
    _verdict(m, 35.0605, measurement_id="meas-1")
    assert m._attempt_fullpipe_verdict()["own"] is True
    _verdict(m, 34.9909, measurement_id="meas-2")
    nxt = m._attempt_fullpipe_verdict()
    assert nxt["own"] is True and nxt["ms"] == 34.9909


def test_the_same_ms_measured_twice_is_two_measurements(mcp):
    """After a revert the pipeline legitimately measures the same number again. Identical ms must not
    be mistaken for a re-read when the measurement really did run."""
    m = mcp
    _verdict(m, 35.0605, measurement_id="meas-1")
    assert m._attempt_fullpipe_verdict()["own"] is True
    _verdict(m, 35.0605, measurement_id="meas-2")
    assert m._attempt_fullpipe_verdict()["own"] is True


def test_a_verdict_with_no_measurement_id_is_never_owned(mcp):
    """Fail CLOSED. A verdict from a path that does not stamp an id has not been shown to correspond
    to a trace replay, and a fabricated win is the one outcome that cannot be corrected afterwards."""
    m = mcp
    _verdict(m, 35.0605)  # no id
    v = m._attempt_fullpipe_verdict()
    assert v["own"] is False and v["win"] is False


# ---------------------------------------------------------------- the verdict must still be honest


def test_a_slower_measurement_is_owned_but_is_not_a_win(mcp):
    """Ownership and winning are separate. Hiding a regression would be the worse bug."""
    m = mcp
    _verdict(m, 35.0605, measurement_id="meas-1")
    m._attempt_fullpipe_verdict()
    _verdict(m, 36.5, measurement_id="meas-2", best_ms=35.0605)
    v = m._attempt_fullpipe_verdict()
    assert v["own"] is True and v["win"] is False and v["delta"] > 0


def test_a_non_ok_status_is_never_owned(mcp):
    m = mcp
    m.record_gate_verdict("full_pipeline", "diverged", full_pipeline_ms=99.0, measurement_id="meas-x")
    assert m._attempt_fullpipe_verdict()["own"] is False


def test_a_zero_or_missing_ms_is_never_owned(mcp):
    m = mcp
    for bad in (0, -1, None):
        m.record_gate_verdict("full_pipeline", "ok", full_pipeline_ms=bad, measurement_id="meas-y")
        assert m._attempt_fullpipe_verdict()["own"] is False, bad


def test_the_consumed_marker_survives_a_reload(mcp, tmp_path, monkeypatch):
    """The MCP server is a long-lived process but the marker must not live only in memory -- a
    restart mid-run would re-open every measurement for re-claiming."""
    m = mcp
    _verdict(m, 35.0605, measurement_id="meas-1")
    assert m._attempt_fullpipe_verdict()["own"] is True
    import models.experimental.perf_automation.cc_optimize.perf_mcp as again

    importlib.reload(again)
    assert again._attempt_fullpipe_verdict()["own"] is False


def test_the_marker_is_written_where_the_state_dir_says(mcp, tmp_path):
    m = mcp
    _verdict(m, 35.0605, measurement_id="meas-1")
    m._attempt_fullpipe_verdict()
    assert str(m._consumed_verdict_path()).startswith(str(tmp_path))
    assert json.loads(m._consumed_verdict_path().read_text())
