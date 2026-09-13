"""A stage's win is measured against how much THAT stage's reading wobbles.

_FULLPIPE_TOL was chosen in July 2026 for one quantity -- the whole-pipeline number -- and inherited
in August when the per-stage improved/regressed test was added. So every stage had to clear the
headline's spread.

On voxtral_mini_3b_2507 (2026-09-04) that cost real work. Decode repeats to 0.04%; prefill moves
several percent. Prefill's two genuine wins that run were -6.5% and -5.0%, both under the inherited
8%, so each was recorded as "not improved" and survived only because encode cleared 8% in the same
measurement and carried them. Whether a real gain counted came down to what else happened to be
running -- and prefill needs about 29 such gains to reach its band.

The samples were already being taken (_FULLPIPE_SAMPLES = 3), and the read set beside these timings
already refuses to be pinned when its own samples disagree. The evidence was in hand; only the
timings ignored it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
_CC = PERF / "cc_optimize"
for _p in (str(PERF), str(PERF.parent.parent.parent), str(_CC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SRC = (_CC / "perf_mcp.py").read_text(encoding="utf-8")


def _pm():
    from cc_optimize import perf_mcp

    return perf_mcp


def _pm_isolated(monkeypatch, tmp_path):
    """A module whose state dir is this test's, loaded the way the other round-trip tests load it."""
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(tmp_path / "m"))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-under-test")
    (tmp_path / "m").mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("pmcp_stage_spread", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_a_quiet_stage_keeps_a_small_win():
    """The measurement repeats to 1%, so a 5% gain is a result -- the old bar called it nothing."""
    m = _pm()
    d = m._stage_deltas({"s": 152.4}, {"s": 160.6}, {"s": 0.01})
    assert d["s"]["improved"] is True
    assert d["s"]["regressed"] is False


def test_the_same_win_is_refused_when_that_stage_is_noisy():
    """Identical numbers, spread of 12%: now it is inside the noise and must not count."""
    m = _pm()
    d = m._stage_deltas({"s": 152.4}, {"s": 160.6}, {"s": 0.12})
    assert d["s"]["improved"] is False


def test_each_stage_is_judged_separately():
    """One quiet stage and one noisy one, the same percentage change, different verdicts."""
    m = _pm()
    d = m._stage_deltas({"a": 95.0, "b": 95.0}, {"a": 100.0, "b": 100.0}, {"a": 0.005, "b": 0.20})
    assert d["a"]["improved"] is True
    assert d["b"]["improved"] is False


def test_a_stage_that_states_no_spread_still_gets_judged():
    """No evidence is not a licence to accept anything: the inherited constant remains the floor."""
    m = _pm()
    tol = m._FULLPIPE_TOL
    inside = {"s": 100.0 * (1.0 - tol / 2.0)}
    outside = {"s": 100.0 * (1.0 - tol * 2.0)}
    assert m._stage_deltas(inside, {"s": 100.0}, {})["s"]["improved"] is False
    assert m._stage_deltas(outside, {"s": 100.0}, {})["s"]["improved"] is True


def test_the_bar_each_stage_was_held_to_is_reported():
    """A verdict whose threshold cannot be seen cannot be argued with."""
    m = _pm()
    d = m._stage_deltas({"s": 99.0}, {"s": 100.0}, {"s": 0.02})
    assert d["s"]["tol_pct"] == 2.0


def test_one_reading_states_no_spread():
    """A single sample has no wobble to report, and zero would read as a perfect measurement."""
    m = _pm()
    assert m._sample_spread([]) is None
    assert m._sample_spread([12.0]) is None
    assert m._sample_spread([10.0, 11.0, 12.0]) == (12.0 - 10.0) / 11.0


def test_the_spread_survives_the_round_trip(monkeypatch, tmp_path):
    """It is written beside the timings it belongs to and read back through the same doc."""
    m = _pm_isolated(monkeypatch, tmp_path)
    m._persist_stage_ms({"s": 10.0}, None, None, None, 0, 0, None, None, {"s": 0.031})
    assert m.read_stage_spread().get("s") == 0.031


def test_the_spread_is_computed_in_exactly_one_place():
    """The read set and the timings ask the same question; two answers is how they drift apart."""
    assert _SRC.count("def _sample_spread") == 1
    assert "(max(_svals) - min(_svals))" not in _SRC, "a second, inline spread calculation came back"


def test_no_stage_name_is_typed_into_the_judgement():
    """Stages come from the capture; a name written here would outlive the model that had it."""
    i = _SRC.index("def _stage_deltas")
    seg = _SRC[i : i + 2000]
    body = seg[seg.index('"""', seg.index('"""') + 3) :]
    for typed in ("decode", "prefill", "encode"):
        assert '"%s"' % typed not in body, typed
