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
    monkeypatch.setattr(perf_mcp, "_LAST_TARGET", {"op": "MatmulDeviceOperation 64 x 2048 x 4096", "rung": "knob:dtype"})
    perf_mcp._record_committed_win("perf(attn): bf8_b weights on qkv+o_proj")
    assert len(recs) == 1
    r = recs[0]
    assert r["beat_baseline"] is True
    assert r["op_signature"] == "MatmulDeviceOperation 64 x 2048 x 4096"
    assert r["kernel_kind"] == "dtype"  # 'knob:' prefix stripped -> ladder column
    assert r["note"].startswith("committed:")


def test_git_commit_records_win_on_success(monkeypatch):
    recs = _capture_appends(monkeypatch)
    monkeypatch.setattr(perf_mcp, "_LAST_TARGET", {"op": "MatmulDeviceOperation", "rung": "grid"})
    monkeypatch.setattr(perf_mcp.gitio, "commit", lambda *a, **k: "sha1234")
    monkeypatch.setattr(perf_mcp.gitio, "repo_root", lambda p: perf_mcp._MODEL_ROOT)
    out = _git_commit("perf: full grid")
    assert out["committed"] is True and out["sha"] == "sha1234"
    assert len(recs) == 1 and recs[0]["beat_baseline"] is True


def test_git_commit_no_win_when_commit_fails(monkeypatch):
    recs = _capture_appends(monkeypatch)
    monkeypatch.setattr(perf_mcp, "_LAST_TARGET", {"op": "MatmulDeviceOperation", "rung": "grid"})
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
