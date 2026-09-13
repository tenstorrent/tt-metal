# SPDX-License-Identifier: Apache-2.0
"""git_commit must log the banked lever as a win, so RUN_REPORT.md shows ✓win.

The ✓win marks come only from record_kernel_attempt(beat_baseline=true); the agent often
records the follow-up no-gain re-measurements and never marks the winning moment, so
committed wins render as ·try. git_commit now derives the win mark from the commit itself.
"""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_cw",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"),
)
perf_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_mcp)

_git_commit = getattr(perf_mcp.git_commit, "fn", perf_mcp.git_commit)


def _capture_appends(monkeypatch):
    recs = []
    monkeypatch.setattr(perf_mcp, "_append_attempt", lambda rec: recs.append(rec) or [rec])
    return recs


def test_record_committed_win_marks_current_target(monkeypatch):
    recs = _capture_appends(monkeypatch)
    monkeypatch.setattr(
        perf_mcp,
        "_LAST_TARGET",
        {"op": "MatmulDeviceOperation 64 x 2048 x 4096", "rung": "knob:dtype", "measured_ms": 12.34},
    )
    perf_mcp._record_committed_win("perf(attn): bf8_b weights on qkv+o_proj")
    assert len(recs) == 1
    r = recs[0]
    assert r["beat_baseline"] is True
    assert r["op_signature"] == "MatmulDeviceOperation 64 x 2048 x 4096"
    assert r["kernel_kind"] == "dtype"  # 'knob:' prefix stripped -> ladder column
    assert r["note"].startswith("committed:")


def _gates_ok(monkeypatch, tmp_path, *, ms=17.0, best=18.0, pcc="ok"):
    """Record the two gate verdicts a banked win now REQUIRES."""
    monkeypatch.setattr(perf_mcp, "_gate_verdict_path", lambda: tmp_path / "verdicts.json")
    perf_mcp.record_gate_verdict("pcc", pcc, pcc=0.99)
    perf_mcp.record_gate_verdict("full_pipeline", "ok", full_pipeline_ms=ms, best_ms=best, method="trace")


def test_git_commit_records_win_on_success(monkeypatch, tmp_path):
    """A win is now banked only when BOTH gates recorded ok and the end-to-end best actually moved."""
    recs = _capture_appends(monkeypatch)
    _gates_ok(monkeypatch, tmp_path)
    monkeypatch.setattr(perf_mcp, "_LAST_TARGET", {"op": "MatmulDeviceOperation", "rung": "grid", "measured_ms": 12.34})
    monkeypatch.setattr(perf_mcp.gitio, "commit", lambda *a, **k: "sha1234")
    monkeypatch.setattr(perf_mcp.gitio, "repo_root", lambda p: perf_mcp._MODEL_ROOT)
    out = _git_commit("perf: full grid")
    assert out["committed"] is True and out["sha"] == "sha1234"
    assert len(recs) == 1 and recs[0]["beat_baseline"] is True


def test_git_commit_is_REFUSED_when_no_gate_has_run(monkeypatch, tmp_path):
    """The rule was prose in the docstring and the body committed unconditionally, so a regressed
    end-to-end and a banked win could coexist: d54438bb4b and 7fac4ae685 landed as wins while the
    gate's best had not moved for 20 minutes. An unrun gate is not a passed gate."""
    recs = _capture_appends(monkeypatch)
    monkeypatch.setattr(perf_mcp, "_gate_verdict_path", lambda: tmp_path / "none.json")
    monkeypatch.setattr(perf_mcp.gitio, "commit", lambda *a, **k: "shaXXXX")
    monkeypatch.setattr(perf_mcp.gitio, "repo_root", lambda p: perf_mcp._MODEL_ROOT)
    out = _git_commit("perf: ungated")
    assert out["committed"] is False and "has not run" in out["refused"]
    assert recs == []


def test_git_commit_is_REFUSED_when_the_end_to_end_regressed(monkeypatch, tmp_path):
    recs = _capture_appends(monkeypatch)
    monkeypatch.setattr(perf_mcp, "_gate_verdict_path", lambda: tmp_path / "v.json")
    perf_mcp.record_gate_verdict("pcc", "ok", pcc=0.99)
    perf_mcp.record_gate_verdict("full_pipeline", "regressed", full_pipeline_ms=17.07, best_ms=17.05, method="trace")
    monkeypatch.setattr(perf_mcp.gitio, "commit", lambda *a, **k: "shaYYYY")
    monkeypatch.setattr(perf_mcp.gitio, "repo_root", lambda p: perf_mcp._MODEL_ROOT)
    out = _git_commit("perf: slower end to end")
    assert out["committed"] is False and "regressed" in out["refused"]
    assert recs == []


def test_git_commit_is_REFUSED_when_pcc_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(perf_mcp, "_gate_verdict_path", lambda: tmp_path / "v.json")
    perf_mcp.record_gate_verdict("pcc", "pcc_low", pcc=0.61)
    perf_mcp.record_gate_verdict("full_pipeline", "ok", full_pipeline_ms=17.0, best_ms=18.0, method="trace")
    monkeypatch.setattr(perf_mcp.gitio, "commit", lambda *a, **k: "shaZZZZ")
    monkeypatch.setattr(perf_mcp.gitio, "repo_root", lambda p: perf_mcp._MODEL_ROOT)
    out = _git_commit("perf: broke correctness")
    assert out["committed"] is False and "pcc_low" in out["refused"]


def test_holding_steady_is_acceptable_but_is_NOT_a_win(monkeypatch, tmp_path):
    """status ok includes 'no worse'. Keeping such an edit is fine; crediting it as a win is how 29
    device_ms new-bests became 29 ticks while the end-to-end best moved far fewer times."""
    monkeypatch.setattr(perf_mcp, "_gate_verdict_path", lambda: tmp_path / "v.json")
    perf_mcp.record_gate_verdict("pcc", "ok", pcc=0.99)
    perf_mcp.record_gate_verdict("full_pipeline", "ok", full_pipeline_ms=17.05, best_ms=17.05, method="trace")
    assert perf_mcp.gates_allow_banking()[0] is True
    assert perf_mcp.gate_set_new_best() is False


def test_git_commit_no_win_when_commit_fails(monkeypatch):
    recs = _capture_appends(monkeypatch)
    monkeypatch.setattr(perf_mcp, "_LAST_TARGET", {"op": "MatmulDeviceOperation", "rung": "grid", "measured_ms": 12.34})
    monkeypatch.setattr(perf_mcp.gitio, "commit", lambda *a, **k: "")
    monkeypatch.setattr(perf_mcp.gitio, "repo_root", lambda p: perf_mcp._MODEL_ROOT)
    out = _git_commit("perf: nothing staged")
    assert out["committed"] is False
    assert recs == []  # no commit -> no win record


def test_record_committed_win_noop_without_target(monkeypatch):
    recs = _capture_appends(monkeypatch)
    monkeypatch.setattr(perf_mcp, "_LAST_TARGET", {})
    monkeypatch.setattr(perf_mcp, "_LAST_TARGET_PATH", Path("/nonexistent/xyz.target"))
    perf_mcp._record_committed_win("perf: x")
    assert recs == []  # no target -> nothing recorded, never raises


def test_record_committed_win_carries_the_commit_sha(monkeypatch):
    """The win row names the commit that banked it, so the live dashboard's history table can point
    at the exact sha instead of matching commit-message text after the fact."""
    recs = _capture_appends(monkeypatch)
    monkeypatch.setattr(
        perf_mcp,
        "_LAST_TARGET",
        {"op": "MatmulDeviceOperation 64 x 2048 x 4096", "rung": "knob:dtype", "measured_ms": 12.34},
    )
    perf_mcp._record_committed_win("perf(attn): bf8_b weights", "a3f9c21deadbeef")
    assert recs[0]["commit"] == "a3f9c21deadbeef"
    # legacy call shape (message only) still works and records no sha
    recs.clear()
    perf_mcp._record_committed_win("perf(attn): bf8_b weights")
    assert recs[0]["commit"] is None


def test_git_commit_passes_the_sha_into_the_win_record(monkeypatch, tmp_path):
    recs = _capture_appends(monkeypatch)
    _gates_ok(monkeypatch, tmp_path)
    monkeypatch.setattr(perf_mcp, "_LAST_TARGET", {"op": "MatmulDeviceOperation", "rung": "grid", "measured_ms": 12.34})
    monkeypatch.setattr(perf_mcp.gitio, "commit", lambda *a, **k: "sha1234")
    monkeypatch.setattr(perf_mcp.gitio, "repo_root", lambda p: perf_mcp._MODEL_ROOT)
    out = _git_commit("perf: full grid")
    assert out["committed"] is True
    assert recs[0]["commit"] == "sha1234"
