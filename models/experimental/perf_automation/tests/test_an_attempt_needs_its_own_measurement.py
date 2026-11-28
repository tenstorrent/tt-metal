"""A recorded attempt must carry an end-to-end measurement of its own.

The IRON RULE in the agent prompt is "a real win = check_pcc ok AND check_full_pipeline_latency
status 'ok' AND ...", and gates_allow_banking enforces exactly that at COMMIT time -- absent verdicts
are refused, not assumed. Nothing enforced it at the ATTEMPT boundary: record_kernel_attempt had a
single exit and no refusal path, so a row could be written with fullpipe_ms=None and render `n/m`.

On gemma-3-12b-it that produced 79 of 94 attempts with no end-to-end number at all, and -- before the
ownership fix -- 13 rows that inherited someone else's measurement and were marked wins. The device
side was never the problem: measure_candidate runs for every attempt. What was missing is the replay
that decides whether the change moved the metric the run is scored on.

So the same rule now applies one step earlier: no owned verdict, no recorded attempt.

The exemptions are the cases where a measurement is impossible rather than merely skipped:
  wedged      the candidate crashed or hung the device -- there is nothing to measure
  no target   nothing is being optimized, so nothing is being claimed
An explicit env override stays for reruns of an interrupted session, where the verdict belongs to a
previous process.
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
    # AND THE KERNEL LOG, which is the one these tests actually WRITE. _KERNEL_LOG_PATH is resolved
    # AT IMPORT as `PERF_MCP_KERNEL_LOG or state_dir()/...`, so redirecting the state dir moves it
    # only while that variable is unset -- true in a terminal, false in every real run, where it
    # points at the live ladder. This fixture intends isolation; leaving one path pointing at a
    # run's own state means these tests could append to the ladder the run resumes from, and read
    # 142 real attempts into assertions written for the two they set up.
    monkeypatch.setenv("PERF_MCP_KERNEL_LOG", str(tmp_path / "kernel_attempts.json"))
    monkeypatch.delenv("PERF_MCP_ALLOW_UNMEASURED_ATTEMPT", raising=False)
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _measured(m, ms=35.0605, mid="meas-1", best=None):
    extra = {"full_pipeline_ms": ms, "sha": "", "measurement_id": mid}
    if best is not None:
        extra["best_ms"] = best
    m.record_gate_verdict("full_pipeline", "ok", **extra)


def _record(m, **kw):
    args = dict(op_signature="Matmul 32 x 3840 x 15360", kernel_kind="grid", measured_ms=400.4, beat_baseline=False)
    args.update(kw)
    return m.record_kernel_attempt(**args)


# ---------------------------------------------------------------- the rule


def test_an_attempt_with_no_end_to_end_is_refused(mcp):
    """The 79-of-94 case. No replay, no row."""
    out = _record(mcp)
    assert out.get("recorded") is False, out
    assert "full_pipeline" in str(out.get("refused", "")).lower() or "measure" in str(out.get("refused", "")).lower()


def test_an_attempt_that_owns_a_measurement_is_recorded(mcp):
    m = mcp
    _measured(m)
    out = _record(m)
    assert out.get("recorded") is not False, out


def test_a_second_attempt_on_the_same_measurement_is_refused(mcp):
    """Ownership is single-use -- see test_a_win_belongs_to_the_attempt_that_measured_it.py. The
    second attempt has no measurement of its own, so it is exactly the case this rule refuses."""
    m = mcp
    _measured(m)
    assert _record(m).get("recorded") is not False
    assert _record(m).get("recorded") is False


def test_each_attempt_needs_a_fresh_replay(mcp):
    m = mcp
    for i in range(3):
        _measured(m, ms=35.0 - i * 0.01, mid="meas-%d" % i)
        assert _record(m).get("recorded") is not False, i
        assert _record(m).get("recorded") is False, i


# ---------------------------------------------------------------- exemptions


def test_a_wedged_attempt_is_still_recorded(mcp):
    """A candidate that crashed or hung the device cannot be measured, and losing that record would
    hide the crash and invite the next run to re-derive it."""
    out = _record(mcp, note="wedged/crashed when tried: TT_FATAL ...")
    assert out.get("recorded") is not False, out


def test_the_override_lets_an_interrupted_session_record(mcp, monkeypatch):
    """A resumed run's verdict belongs to the previous process. The escape hatch is explicit and
    named, like PERF_MCP_ALLOW_UNGATED_COMMIT on the commit gate."""
    monkeypatch.setenv("PERF_MCP_ALLOW_UNMEASURED_ATTEMPT", "1")
    assert _record(mcp).get("recorded") is not False


# ---------------------------------------------------------------- what the refusal must not do


def test_a_refusal_says_what_to_do(mcp):
    """The agent reads this string and has to be able to act on it."""
    out = _record(mcp)
    why = str(out.get("refused", ""))
    assert "check_full_pipeline_latency" in why, why


def test_a_refusal_does_not_write_a_row(mcp, tmp_path):
    """A refused attempt that still lands in the kernel log is worse than no rule at all."""
    import json

    m = mcp
    _record(m)
    p = Path(m._kernel_log_path()) if hasattr(m, "_kernel_log_path") else None
    if p is None or not p.exists():
        return
    assert json.loads(p.read_text()) == []


def test_a_refusal_is_not_an_exception(mcp):
    """This runs inside a live agent loop; raising would end the round instead of redirecting it."""
    out = _record(mcp)
    assert isinstance(out, dict)
