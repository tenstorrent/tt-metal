"""_record_committed_win's guard was always right; nothing ever fed it what it asked for.

"A commit is not a measurement" (its own docstring) is the correct rule -- on llama3_1_8b_p150 it
was written specifically because marking every successful git_commit as a win produced 47 fake wins
out of 73, from housekeeping commits with no lever behind them. The guard reads
_load_target().get("measured_ms") before it will bank a win. Traced end to end: _persist_target is
called from exactly one place (termination_check, with next_target's ten ladder fields --
op/op_class/grid/bound_by/rung/gap_ms/reason/stage/prev_op/next_op) and measure_candidate never
touches the target at all. So measured_ms was never on the target, on any commit, on any model --
not intermittently, structurally always absent. Confirmed live: nvidia_nemotron_3_5_lightning's
persisted target file carried exactly those ten keys and no measured_ms; a real run that recorded
30 committed wins on voxtral produced zero rows carrying a commit marker.

record_kernel_attempt already receives measured_ms as its own argument, one call before a
git_commit for the same target. This stamps it onto the SAME target object record_kernel_attempt
just worked with, so a git_commit that follows can finally see it -- without loosening the guard
that correctly rejects a measurement-less commit.

Commit rows becoming real (instead of silently never appearing) exposes a second, previously
theoretical risk: _record_committed_win's row carries measured_ms and kernel_detected_in_source:
True, which is exactly what _rung_allowance and _op_ladder_status count as a try. Without an
exclusion, a real win would burn its rung's retry budget twice -- once for the attempt, once for
saving it. Fixed in the same change: commit_record marks the row, and both counters skip it.
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
    monkeypatch.setenv("PERF_MCP_ALLOW_UNMEASURED_ATTEMPT", "1")
    monkeypatch.delenv("PERF_MCP_ALLOW_RETRIED_RUNG", raising=False)
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    monkeypatch.setattr(m, "_MODEL_ROOT", tmp_path)
    return m


def _next_target_shaped(op="MatmulDeviceOperation 32 x 2688 x 131072", rung="knob:grid"):
    """Exactly the ten fields termination_check's next_target carries -- no measured_ms, because
    the real one never has one either."""
    return {
        "op": op,
        "op_class": "matmul",
        "grid": "partial",
        "bound_by": "memory",
        "rung": rung,
        "gap_ms": 230.06,
        "reason": "no program_config; full-grid available",
        "stage": "decode",
        "prev_op": "TypecastDeviceOperation",
        "next_op": "BinaryNgDeviceOperation",
    }


def _rows(m, path=None):
    p = Path(path or m._KERNEL_LOG_PATH)
    return json.loads(p.read_text()) if p.exists() else []


# ---------------------------------------------------------------- _stamp_target_measurement itself


def test_it_adds_the_measurement_without_losing_the_other_fields(mcp):
    m = mcp
    m._persist_target(_next_target_shaped())
    m._stamp_target_measurement(794.85)
    t = m._load_target()
    assert t["measured_ms"] == 794.85
    assert t["op"] == "MatmulDeviceOperation 32 x 2688 x 131072"
    assert t["rung"] == "knob:grid"


def test_it_persists_to_disk_not_only_memory(mcp):
    """A git_commit call may land in a different process than the one that measured -- the MCP
    server per tool call is not guaranteed to be the same interpreter -- so in-memory alone is not
    enough; _persist_target's own disk write is what _load_target falls back to."""
    m = mcp
    m._persist_target(_next_target_shaped())
    m._stamp_target_measurement(605.12)
    m._LAST_TARGET.clear()  # simulate a fresh process with no in-memory target
    assert m._load_target().get("measured_ms") == 605.12


def test_it_is_a_no_op_with_no_current_target(mcp):
    m = mcp
    assert m._load_target() == {}
    m._stamp_target_measurement(100.0)  # must not raise, must not fabricate a target
    assert m._load_target() == {}


# ---------------------------------------------------------------- the real gap, end to end


def test_record_kernel_attempt_bridges_the_measurement_to_the_target(mcp):
    """The actual bug: termination_check's next_target never carries measured_ms, and nothing
    else ever added it -- so _record_committed_win's guard could never pass. This is the fix."""
    m = mcp
    m._persist_target(_next_target_shaped())
    assert m._load_target().get("measured_ms") is None  # the gap, confirmed present before the fix
    m.record_kernel_attempt("MatmulDeviceOperation 32 x 2688 x 131072", "grid", 794.85, True, note="full-grid config")
    assert m._load_target().get("measured_ms") == 794.85  # bridged


def test_a_commit_right_after_can_now_bank_the_win(mcp, monkeypatch):
    m = mcp
    m._persist_target(_next_target_shaped())
    m.record_kernel_attempt("MatmulDeviceOperation 32 x 2688 x 131072", "grid", 794.85, True, note="full-grid config")
    monkeypatch.setattr(m, "gates_allow_banking", lambda: (True, ""))
    monkeypatch.setattr(m.gitio, "repo_root", lambda p: m._MODEL_ROOT)
    monkeypatch.setattr(m.gitio, "commit", lambda *a, **k: "sha_real")
    out = m.git_commit("perf: full-grid lm_head matmul")
    assert out["committed"] is True and out["sha"] == "sha_real"
    rows = _rows(m)
    commit_rows = [r for r in rows if r.get("commit_record")]
    assert len(commit_rows) == 1
    assert commit_rows[0]["beat_baseline"] is True
    assert commit_rows[0]["commit"] == "sha_real"
    assert commit_rows[0]["measured_ms"] == 794.85


# ---------------------------------------------------------------- commit rows never double-spend a rung


def _history(m, rows):
    Path(m._KERNEL_LOG_PATH).write_text(json.dumps(rows))
    Path(str(m._KERNEL_LOG_PATH) + ".cumulative").write_text(json.dumps(rows))


def _attempt_row(sig, kind, ms):
    return {"op_signature": sig, "kernel_kind": kind, "measured_ms": ms, "kernel_detected_in_source": True}


def _commit_row(sig, kind, ms):
    return {
        "op_signature": sig,
        "kernel_kind": kind,
        "measured_ms": ms,
        "beat_baseline": True,
        "wedged": False,
        "kernel_detected_in_source": True,
        "commit_record": True,
        "commit": "shaZZZZ",
        "note": "committed: perf: win",
    }


def test_a_commit_row_does_not_burn_a_second_try(mcp):
    """One real win, recorded as two rows (attempt + commit), must spend the rung's budget once."""
    m = mcp
    op = "MatmulDeviceOperation 32 x 2688 x 131072"
    _history(m, [_attempt_row(op, "grid", 794.85), _commit_row(op, "grid", 794.85)])
    tries, allowed = m._rung_allowance(op, "grid", m._load_attempts_all())
    assert (tries, allowed) == (1, m._MAX_KNOB_RETRIES)


def test_a_commit_row_still_leaves_a_real_second_attempt_available(mcp):
    op = "MatmulDeviceOperation 32 x 2688 x 131072"
    _history(
        m := mcp,
        [
            _attempt_row(op, "grid", 794.85),
            _commit_row(op, "grid", 794.85),
            _attempt_row(op, "grid", 780.0),
        ],
    )
    tries, allowed = m._rung_allowance(op, "grid", m._load_attempts_all())
    assert tries == 2 and tries >= allowed  # 2 real attempts, cap reached -- not 3
