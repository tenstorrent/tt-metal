# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which store owns the modeled floor -- the OTHER half of test_floor_anchor_writeonce.py.

The reported floor must not move (an optimized re-profile lowering its own target reads as a
regression that never happened: llama3_1_8b_p150 rendered 83% -> 55% at-floor while measuring
FASTER). That pin lives in the measurement ledger, and these tests assert the split that makes it
work:

    throughput snapshot  = the CURRENT build          -> refreshes every profile, by design
    measurement ledger   = the anchors the report uses -> written once, never moved

This file previously asserted the snapshot did the pinning too. Two stores meant two answers to
"what is the target", and the snapshot copy could not be corrected mid-run, because the MCP server
loads perf_mcp once at startup -- so a fix to it took effect a round late while the ledger-side fix
took effect on the next render.
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
    spec = importlib.util.spec_from_file_location("pm_floor_ut", _ROOT / "cc_optimize" / "perf_mcp.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pm_floor_ut"] = m
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "_throughput_path", lambda: tmp_path / "tp.json")
    return m


class _T:
    theoretical_rate = 0.0
    band = (0.0, 0.0)
    active_bytes = 0
    tp_degree = 1


def _snapshot(m, monkeypatch, floor, depth="16"):
    monkeypatch.setenv("TT_PERF_LAYERS", depth)
    monkeypatch.setattr(m, "_select_perf_target", lambda rep: (_T(), "model", False))
    m._persist_throughput({"modeled_floor_ms": floor})


def test_snapshot_tracks_the_current_build(tmp_path, monkeypatch):
    """It describes what the model does NOW, so a lower floor after a dtype change belongs here."""
    m = _pm(tmp_path, monkeypatch)
    _snapshot(m, monkeypatch, 537.23)
    _snapshot(m, monkeypatch, 341.47)
    assert json.loads((tmp_path / "tp.json").read_text())["modeled_floor_ms"] == 341.47


def test_snapshot_holds_no_second_pin(tmp_path, monkeypatch):
    """A pin field here would be a second answer to 'what is the target'."""
    m = _pm(tmp_path, monkeypatch)
    _snapshot(m, monkeypatch, 537.23)
    assert "floor_pinned_from" not in json.loads((tmp_path / "tp.json").read_text())


def test_the_other_snapshot_fields_still_refresh(tmp_path, monkeypatch):
    m = _pm(tmp_path, monkeypatch)
    _snapshot(m, monkeypatch, 537.23, depth="16")
    _snapshot(m, monkeypatch, 341.47, depth="2")
    got = json.loads((tmp_path / "tp.json").read_text())
    assert got["perf_layers"] == "2"


def test_a_refreshed_snapshot_cannot_change_the_reported_floor(tmp_path, monkeypatch):
    """END TO END across both stores: the snapshot drops 537 -> 341, the REPORT still says 537."""
    m = _pm(tmp_path, monkeypatch)
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "ledger.jsonl"))
    spec = importlib.util.spec_from_file_location("sm_floor_ut", _ROOT / "cc_optimize" / "summary.py")
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)

    monkeypatch.setattr(m, "_MODEL_ROOT", tmp_path / "m", raising=False)
    _snapshot(m, monkeypatch, 537.23)
    first = json.loads((tmp_path / "tp.json").read_text())
    assert "537.23 ms" in "\n".join(sm._roofline_lines(first, 648.17, None, "m", "main"))

    _snapshot(m, monkeypatch, 341.47)
    later = json.loads((tmp_path / "tp.json").read_text())
    txt = "\n".join(sm._roofline_lines(later, 615.69, None, "m", "main"))
    assert "modeled floor       : 537.23 ms" in txt
    assert "341.47" not in txt, txt
