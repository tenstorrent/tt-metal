# SPDX-License-Identifier: Apache-2.0
"""_last_committed_ms drives the report's 'final' from the last banked win, not the stale
perf_mcp_baseline.json — the +0.0%-despite-real-wins bug. Uses the last (current) win, not the
best-ever, honoring the pin-current design.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

_S = importlib.util.spec_from_file_location(
    "run_lcm", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "run.py"))
run = importlib.util.module_from_spec(_S)
sys.modules["run_lcm"] = run
_S.loader.exec_module(run)


def _log(rows):
    p = tempfile.mktemp(suffix=".json")
    json.dump(rows, open(p, "w"))
    return p


def test_returns_last_banked_win():
    # wins 31 -> 28 -> 18.8 ; current committed = the LAST one (18.8), not min-by-accident
    log = _log([
        {"kernel_kind": "grid", "measured_ms": 31.2, "beat_baseline": True},
        {"kernel_kind": "dtype", "measured_ms": 28.4, "beat_baseline": True},
        {"kernel_kind": "shard", "measured_ms": 18.8, "beat_baseline": True},
        {"kernel_kind": "cpp", "measured_ms": 40.0, "beat_baseline": False},  # no-gain after -> ignored
    ])
    assert run._last_committed_ms(log) == 18.8


def test_pin_current_uses_last_not_best():
    # a better win (15) recorded, then a later win at 20 -> current is 20, not the best 15
    log = _log([
        {"kernel_kind": "dtype", "measured_ms": 15.0, "beat_baseline": True},
        {"kernel_kind": "shard", "measured_ms": 20.0, "beat_baseline": True},
    ])
    assert run._last_committed_ms(log) == 20.0  # pin current (last), not best-ever


def test_none_when_no_wins():
    log = _log([{"kernel_kind": "grid", "measured_ms": 50.0, "beat_baseline": False}])
    assert run._last_committed_ms(log) is None  # -> caller falls back to _baseline_ms()


def test_none_on_bad_path():
    assert run._last_committed_ms("/nonexistent/x.json") is None
