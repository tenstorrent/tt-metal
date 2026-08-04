"""One reading cannot grade a win smaller than the board's spread.

`check_full_pipeline_latency` took a single measurement and compared it to the committed best:

    ms, method, err, path = _run_full_pipeline_ms()      # perf_mcp.py:2889

That one reading decided every verdict, and the board does not repeat itself. Six interleaved runs of
the SAME code on gemma-3-12b-it, alternating a no-op edit so thermal drift hit both arms equally:

    33.34   34.72   35.19   35.21   35.29   35.62      spread 2.28 ms  (28.1 - 30.0 tok/s/u)

The wins the ladder hunts are 0.7-1.3 ms. Against that spread a single reading is a coin flip in BOTH
directions -- a neutral lever reads as a 1 ms win, a real 1 ms win reads as a loss -- and either way
the result is written as CONCLUSIVE, so the ladder never revisits it.

That is the story of this model, not a hypothetical. 145 attempts produced one "win" of -0.79 ms;
re-measured later the same lever read +0.32. And the bar itself was one lucky cold sample (33.9011)
that nothing warm could beat, so 144 attempts were graded against a number the board could not
reproduce.

MEDIAN, not mean. This bench emits 55-66 ms readings (a real regression left applied across
attempts), and a mean would let one of those invent or destroy a win outright; the median ignores it
so long as most readings are sane.

The spread is returned alongside, because a delta smaller than the spread it came from is not a
verdict and the caller should be able to see that.

Default 3. PERF_MCP_GATE_REPS=1 restores single-shot exactly, for anyone who wants the old speed and
accepts the coin flip.
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
    monkeypatch.delenv("PERF_MCP_GATE_REPS", raising=False)
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _feed(mcp, monkeypatch, readings):
    """Make _run_full_pipeline_ms return each reading in turn; None means that rep failed."""
    seq, calls = list(readings), {"n": 0}

    def fake():
        v = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        return (None, "trace", "boom", "p") if v is None else (v, "trace", None, "p")

    monkeypatch.setattr(mcp, "_run_full_pipeline_ms", fake)
    return calls


# ---------------------------------------------------------------- it repeats, and takes the middle


def test_it_measures_three_times_by_default(mcp, monkeypatch):
    calls = _feed(mcp, monkeypatch, [35.0, 35.1, 35.2])
    mcp._measure_full_pipeline_median()
    assert calls["n"] == 3, calls


def test_it_returns_the_median_not_the_first(mcp, monkeypatch):
    """The first reading of a run is the one most likely to be unrepresentative."""
    _feed(mcp, monkeypatch, [55.0, 35.1, 35.2])
    ms, *_ = mcp._measure_full_pipeline_median()
    assert ms == 35.2, ms


def test_one_degraded_reading_does_not_decide(mcp, monkeypatch):
    """A mean of [66.5, 35.0, 35.1] is 45.5 and would manufacture a huge fake win."""
    _feed(mcp, monkeypatch, [66.5, 35.0, 35.1])
    ms, *_ = mcp._measure_full_pipeline_median()
    assert ms < 36, ms


def test_the_spread_comes_back_too(mcp, monkeypatch):
    """A delta smaller than the spread is not a verdict; the caller must be able to tell."""
    _feed(mcp, monkeypatch, [35.0, 36.0, 37.0])
    *_rest, spread = mcp._measure_full_pipeline_median()
    assert abs(spread - 2.0) < 1e-6, spread


# ---------------------------------------------------------------- configurable, and reversible


def test_reps_of_one_is_exactly_the_old_behaviour(mcp, monkeypatch):
    monkeypatch.setenv("PERF_MCP_GATE_REPS", "1")
    importlib.reload(mcp)
    calls = _feed(mcp, monkeypatch, [35.0, 99.0, 99.0])
    ms, *_ = mcp._measure_full_pipeline_median()
    assert calls["n"] == 1 and ms == 35.0


def test_reps_can_be_raised(mcp, monkeypatch):
    monkeypatch.setenv("PERF_MCP_GATE_REPS", "5")
    importlib.reload(mcp)
    calls = _feed(mcp, monkeypatch, [35.0] * 5)
    mcp._measure_full_pipeline_median()
    assert calls["n"] == 5


def test_an_even_count_averages_the_middle_two(mcp, monkeypatch):
    monkeypatch.setenv("PERF_MCP_GATE_REPS", "4")
    importlib.reload(mcp)
    _feed(mcp, monkeypatch, [35.0, 36.0, 38.0, 39.0])
    ms, *_ = mcp._measure_full_pipeline_median()
    assert abs(ms - 37.0) < 1e-6, ms


# ---------------------------------------------------------------- failures


def test_every_rep_failing_still_reports_the_error(mcp, monkeypatch):
    """No fabricated number when nothing measured."""
    _feed(mcp, monkeypatch, [None, None, None])
    ms, _method, err, _path, _spread = mcp._measure_full_pipeline_median()
    assert ms is None and err


def test_one_failed_rep_does_not_discard_the_others(mcp, monkeypatch):
    """Throwing away two good readings because one replay flaked would be worse than measuring once."""
    _feed(mcp, monkeypatch, [None, 35.0, 35.2])
    ms, *_ = mcp._measure_full_pipeline_median()
    assert ms is not None and 34 < ms < 36, ms
