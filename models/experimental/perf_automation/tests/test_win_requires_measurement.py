# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A ✓win must mean "measured faster and committed", never merely "a commit happened".

_record_committed_win set beat_baseline=True on EVERY successful git_commit, with measured_ms taken
from the target -- often None. The agent uses git_commit for housekeeping too, so
"refresh the generated RUN_REPORT", "checkpoint the perf test" and a comment-only
"record the measured dead ends" all rendered as wins. On llama3_1_8b_p150 that was 47 of 73 wins in
one run, and it put a ✓ in the fidelity column while both real fidelity measurements showed no gain.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _pm(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text('{"config": {}, "perf_test_resolved": {"path": "t.py"}}')
    monkeypatch.setenv("PERF_MCP_KERNEL_LOG", str(tmp_path / "kl.json"))
    spec = importlib.util.spec_from_file_location("pm_win_ut", _ROOT / "cc_optimize" / "perf_mcp.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pm_win_ut"] = m
    spec.loader.exec_module(m)
    return m


def _summary():
    spec = importlib.util.spec_from_file_location("sm_win_ut", _ROOT / "cc_optimize" / "summary.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["sm_win_ut"] = m
    spec.loader.exec_module(m)
    return m


def test_an_unmeasured_commit_is_not_recorded_as_a_win(tmp_path, monkeypatch):
    """THE REGRESSION: a comment-only commit became a ✓win."""
    pm = _pm(tmp_path, monkeypatch)
    monkeypatch.setattr(pm, "_load_target", lambda: {"op": "MatmulDeviceOperation", "rung": "knob:fidelity"})
    pm._record_committed_win("record the measured dead ends  Comment-only.")
    rows = json.loads((tmp_path / "kl.json").read_text()) if (tmp_path / "kl.json").exists() else []
    assert not [r for r in rows if r.get("beat_baseline")], rows


def test_a_measured_commit_is_still_recorded_as_a_win(tmp_path, monkeypatch):
    """The original intent must survive: a genuinely banked lever still gets its ✓."""
    pm = _pm(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pm, "_load_target", lambda: {"op": "MatmulDeviceOperation", "rung": "knob:dtype", "measured_ms": 648.17}
    )
    pm._record_committed_win("put the LM head weight on bf4_b")
    rows = json.loads((tmp_path / "kl.json").read_text())
    wins = [r for r in rows if r.get("beat_baseline")]
    assert len(wins) == 1 and wins[0]["measured_ms"] == 648.17


def test_the_renderer_refuses_an_unmeasured_win_from_an_old_log(tmp_path):
    """Logs already on disk carry these rows, so the renderer must refuse them too -- otherwise every
    previously written kernel log keeps producing inflated ✓ marks."""
    sm = _summary()
    kl = tmp_path / "kl.json"
    kl.write_text(
        json.dumps(
            [
                {"op_signature": "Matmul", "kernel_kind": "fidelity", "measured_ms": None, "beat_baseline": True},
                {"op_signature": "Matmul", "kernel_kind": "fidelity", "measured_ms": 664.13, "beat_baseline": False},
            ]
        )
    )
    out = sm.render_summary(kl, model="m", task="main", finalized=True)
    row = next(l for l in out.splitlines() if l.startswith("Matmul"))
    assert "✓win" not in row, row
    assert "·try" in row, row


def test_a_measured_win_still_renders_a_tick(tmp_path):
    sm = _summary()
    kl = tmp_path / "kl.json"
    kl.write_text(
        json.dumps([{"op_signature": "Matmul", "kernel_kind": "dtype", "measured_ms": 648.17, "beat_baseline": True}])
    )
    out = sm.render_summary(kl, model="m", task="main", finalized=True)
    row = next(l for l in out.splitlines() if l.startswith("Matmul"))
    assert "✓win" in row, row
