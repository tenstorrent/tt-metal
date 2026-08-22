"""A win is a change that is IN THE TREE. Anything else is an attempt.

record_kernel_attempt set beat_baseline from the end-to-end verdict, at RECORD time -- before the
commit-or-revert decision exists. _record_committed_win then sets it again, from the commit, which is
what the report was always meant to show: "git_commit IS the bank-a-verified-win action ... Deriving
the win mark from the commit itself makes the report reflect what was actually banked."

Two writers for one flag, and the early one fires on things that never reach the tree. On
gemma-3-12b-it, run 21 showed three ✓win marks after 80 minutes with ZERO commits:

  LayerNorm/shard        applied, measured -0.84%, agent REVERTED it as noise against a clean
                         bookend -- claimed_beat_baseline=False, and the gate marked it a win anyway
  RoPE/structural        note "none: ...all are closed" -- no edit at all, device_ms unchanged
  BinaryNg/structural    note "none: the op's cost is NOT its own work" -- no edit, device_ms
                         unchanged at 381.2266

All three "beat" the reference because the run's baseline (36.2548 ms) had been measured on freshly
reset chips and every later reading was warm (~35 ms). The ratchet never advanced -- nothing was
committed -- so the same cold number stayed the bar for every attempt.

So the flag now has ONE writer: the commit. An attempt records what it measured and nothing more.
The verdict fields (fullpipe_ms, delta) are still recorded, because they are evidence; they simply
stop deciding a mark that only a commit can earn.
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
    monkeypatch.setenv("PERF_MCP_ALLOW_UNMEASURED_ATTEMPT", "1")
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    monkeypatch.setattr(m, "_MODEL_ROOT", tmp_path)
    monkeypatch.setattr(m, "_KERNEL_LOG_PATH", tmp_path / "attempts.json")
    return m


def _verdict(m, ms, best, mid):
    m.record_gate_verdict("full_pipeline", "ok", full_pipeline_ms=ms, best_ms=best, sha="", measurement_id=mid)


def _record(m, claim, op="LayerNormDeviceOperation", kind="shard"):
    return m.record_kernel_attempt(op, kind, 378.04, claim, note="measured")


def _rows(m):
    import json

    p = Path(m._KERNEL_LOG_PATH)
    return json.loads(p.read_text()) if p.exists() else []


# ---------------------------------------------------------------- the reported case


def test_a_reverted_change_is_not_a_win(mcp):
    """LayerNorm/shard: applied, measured better than a COLD baseline, agent reverted it as noise.
    claimed=False must not become ✓win."""
    m = mcp
    _verdict(m, 35.4933, 36.2548, "meas-1")
    _record(m, claim=False)
    assert [r["beat_baseline"] for r in _rows(m)] == [False]


def test_an_investigation_that_changed_nothing_is_not_a_win(mcp):
    """RoPE/BinaryNg structural: note 'none: ...', device_ms unchanged, no edit made."""
    m = mcp
    _verdict(m, 34.9585, 36.2548, "meas-2")
    m.record_kernel_attempt("BinaryNgDeviceOperation", "structural", 381.2266, False, note="none: nothing reducible")
    assert [r["beat_baseline"] for r in _rows(m)] == [False]


def test_three_attempts_against_a_cold_baseline_yield_no_wins(mcp):
    """The exact run-21 shape: every warm reading beats a cold 36.2548 and none was committed."""
    m = mcp
    for i, ms in enumerate((35.4933, 35.3165, 34.9585)):
        _verdict(m, ms, 36.2548, "meas-%d" % i)
        m.record_kernel_attempt("Op%d" % i, "structural", 381.2266, False, note="none")
    assert sum(1 for r in _rows(m) if r["beat_baseline"]) == 0


# ---------------------------------------------------------------- the commit is the only writer


def test_a_commit_marks_the_win(mcp, monkeypatch):
    """_record_committed_win appends the ✓win row. That is the one place the mark comes from."""
    m = mcp
    monkeypatch.setattr(
        m, "_load_target", lambda: {"op": "LayerNormDeviceOperation", "rung": "shard", "measured_ms": 378.04}
    )
    m._record_committed_win("gemma3: widen the prefill norm")
    rows = _rows(m)
    assert rows and rows[-1]["beat_baseline"] is True
    assert rows[-1]["note"].startswith("committed:")


def test_a_commit_with_no_measurement_still_marks_nothing(mcp, monkeypatch):
    """A COMMIT IS NOT A MEASUREMENT -- housekeeping commits must not render as wins. This guard
    already existed and must survive."""
    m = mcp
    monkeypatch.setattr(m, "_load_target", lambda: {"op": "X", "rung": "grid", "measured_ms": None})
    m._record_committed_win("refresh the generated report")
    assert not any(r["beat_baseline"] for r in _rows(m))


# ---------------------------------------------------------------- evidence is still recorded


def test_the_measurement_is_still_recorded_on_the_attempt(mcp):
    """Dropping the FLAG must not drop the numbers -- the delta is the evidence a reader needs."""
    m = mcp
    _verdict(m, 35.4933, 36.2548, "meas-1")
    _record(m, claim=True)
    r = _rows(m)[0]
    assert r["fullpipe_ms"] == 35.4933 and r["fullpipe_delta_ms"] is not None
    assert r["claimed_beat_baseline"] is True


def test_an_agent_claim_alone_cannot_mark_a_win(mcp):
    """The claim is recorded for the record, never promoted to the mark."""
    m = mcp
    _verdict(m, 35.4933, 36.2548, "meas-1")
    _record(m, claim=True)
    assert _rows(m)[0]["beat_baseline"] is False
    assert _rows(m)[0]["claimed_beat_baseline"] is True
