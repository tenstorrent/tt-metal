# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The modeled floor is pinned at the baseline, so "% at floor" can converge.

The snapshot write was unconditional, so every re-profile recomputed the floor from the
already-optimized state. Any lever that REMOVES BYTES lowers the measurement and the floor together
(bf8_b -> bf4_b halves a weight read; dropping a 128-token pad quarters the prefill), so the ratio
stands still however much faster the model gets -- the target chases the measurement down.

llama3_1_8b_p150 went 537.23 -> 341.47 ms of floor between runs while measuring FASTER, and rendered
as 83% -> 55% at-floor: a regression that never happened.
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
    theoretical_tok_s = 0.0
    band = (0.0, 0.0)
    active_bytes = 0
    tp_degree = 1


def _snapshot(m, monkeypatch, floor, depth="16"):
    monkeypatch.setenv("TT_PERF_LAYERS", depth)
    monkeypatch.setattr(m, "_select_perf_target", lambda rep: (_T(), "model", False))
    m._persist_throughput({"modeled_floor_ms": floor})


def test_the_floor_is_pinned_at_the_first_profile(tmp_path, monkeypatch):
    """THE REGRESSION: an optimized re-profile must not lower the target it is judged against."""
    m = _pm(tmp_path, monkeypatch)
    _snapshot(m, monkeypatch, 537.23)
    _snapshot(m, monkeypatch, 341.47)  # a later, optimized profile
    got = json.loads((tmp_path / "tp.json").read_text())
    assert got["modeled_floor_ms"] == 537.23, got["modeled_floor_ms"]
    assert got["floor_pinned_from"] == "baseline"


def test_repeated_profiles_never_move_it(tmp_path, monkeypatch):
    m = _pm(tmp_path, monkeypatch)
    _snapshot(m, monkeypatch, 537.23)
    for f in (500.0, 420.0, 341.47, 300.0):
        _snapshot(m, monkeypatch, f)
    assert json.loads((tmp_path / "tp.json").read_text())["modeled_floor_ms"] == 537.23


def test_a_different_depth_re_pins_rather_than_comparing_across_windows(tmp_path, monkeypatch):
    """A floor summed over 2 layers must never be kept for a 16-layer measurement."""
    m = _pm(tmp_path, monkeypatch)
    _snapshot(m, monkeypatch, 80.0, depth="2")
    _snapshot(m, monkeypatch, 537.23, depth="16")
    got = json.loads((tmp_path / "tp.json").read_text())
    assert got["modeled_floor_ms"] == 537.23 and got["perf_layers"] == "16"


def test_the_other_snapshot_fields_still_refresh(tmp_path, monkeypatch):
    """Only the floor is pinned; the rest of the snapshot tracks the current state."""
    m = _pm(tmp_path, monkeypatch)
    _snapshot(m, monkeypatch, 537.23)
    _snapshot(m, monkeypatch, 341.47)
    got = json.loads((tmp_path / "tp.json").read_text())
    assert got["modeled_floor_ms"] == 537.23
    assert "peak_bw_gbps" in got and "band" in got
