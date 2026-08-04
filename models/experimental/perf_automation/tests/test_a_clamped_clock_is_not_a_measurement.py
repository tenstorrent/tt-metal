"""A reading taken while the clock was clamped is not a slower reading. It is not a reading.

Measured on a liquid-cooled p300c running gemma-3-12b-it, IDENTICAL code every time, the only
variable being the die temperature the run STARTED at:

    69.8C -> 35.83 ms   27.91 tok/s/u   AICLK settles at 1350
    73.5C -> 36.75 ms   27.21 tok/s/u   settles, but the NEXT run no longer does
    78.3C -> 58.06 ms   17.22 tok/s/u   UMD: "clamped by max-arbiter index 7 at 800 MHz"
    79.9C -> 69.94 ms   14.30 tok/s/u   same clamp

A 1.9x swing with no code change. Three separate wrong diagnoses were built on top of these numbers
-- a "1.6-1.9x concat-heads regression", thermal throttling at the 110C trip limit, and a board
stuck at 800 MHz -- before the UMD warning turned out to be saying it outright.

Two things follow, and the second is the one that is easy to get wrong:

1. A device reset does NOT fix it. A reset does not cool a board. Both post-reset runs above were
   clamped, and the second was WORSE than the first because the board was hotter by then.

2. "Measure everything hot so at least it is consistent" does NOT work. 58.06 and 69.94 are both
   clamped at 800 MHz and are 20% apart -- the clamped state is not a stable operating point, so
   hot readings are not comparable to each other either.

So the only sound rule is to refuse the reading: wait for headroom before measuring, and if the run
reports its clock was clamped, DISCARD what it measured rather than record it. That turns silently
wrong numbers into fewer, correct ones -- and when the board is simply too hot to measure, into a
loud failure instead of a 68.3 ms baseline anchor.

This is what happened without it: run 30's BEFORE anchor was written as 68.3 ms (14.6 tok/s/u)
against a true 34.99. Every candidate the run went on to measure would have been graded against
that, so anything at all banked as a ~30 ms win, written conclusive.
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
    monkeypatch.delenv("PERF_MCP_THERMAL_GATE", raising=False)
    monkeypatch.setenv("PERF_MCP_THERMAL_POLL_S", "0")
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _feed(mcp, monkeypatch, readings, clamped=()):
    """Each reading is one run; `clamped` marks which run indices report a clamped clock."""
    seq, calls = list(readings), {"n": 0}

    def fake():
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        mcp._LAST_RUN_CLAMPED = i in clamped
        globals_ = getattr(mcp, "__dict__")
        globals_["_LAST_RUN_CLAMPED"] = i in clamped
        v = seq[i]
        return (None, "trace", "boom", "p") if v is None else (v, "trace", None, "p")

    monkeypatch.setattr(mcp, "_run_full_pipeline_ms", fake)
    monkeypatch.setattr(mcp, "_wait_for_thermal_headroom", lambda: (True, 65.0))
    return calls


# ---------------------------------------------------------------- clamped readings are discarded


def test_a_clamped_reading_is_not_used(mcp, monkeypatch):
    """The run-30 case: a 68.3 ms clamped reading must not become the answer."""
    _feed(mcp, monkeypatch, [68.3, 35.0, 35.1, 35.2], clamped={0})
    ms, *_ = mcp._measure_full_pipeline_median()
    assert ms is not None and ms < 36, ms


def test_it_retries_until_it_has_enough_clean_readings(mcp, monkeypatch):
    """Discarding must not simply shrink the sample -- it has to go get another one."""
    calls = _feed(mcp, monkeypatch, [68.3, 69.9, 35.0, 35.1, 35.2], clamped={0, 1})
    mcp._measure_full_pipeline_median()
    assert calls["n"] == 5, calls


def test_an_all_clamped_board_fails_loudly(mcp, monkeypatch):
    """No number at all beats a wrong number: this is what stops a bad BEFORE anchor being written."""
    _feed(mcp, monkeypatch, [68.3] * 12, clamped=set(range(12)))
    ms, _method, err, _path, _spread = mcp._measure_full_pipeline_median()
    assert ms is None and "clamped" in (err or "").lower(), (ms, err)


def test_retries_are_bounded(mcp, monkeypatch):
    """A permanently hot board must terminate, not spin forever."""
    calls = _feed(mcp, monkeypatch, [68.3] * 50, clamped=set(range(50)))
    mcp._measure_full_pipeline_median()
    assert calls["n"] <= mcp._GATE_REPS + mcp._THERMAL_RETRIES, calls


def test_unclamped_readings_are_untouched(mcp, monkeypatch):
    """The healthy path must behave exactly as before the gate existed."""
    _feed(mcp, monkeypatch, [35.0, 35.1, 35.2])
    ms, _m, err, _p, _s = mcp._measure_full_pipeline_median()
    assert err is None and ms == 35.1, (ms, err)


def test_the_gate_can_be_switched_off(mcp, monkeypatch):
    """An escape hatch for a board whose telemetry lies, or a bring-up on unknown silicon.

    Switched off, the clamped reading is KEPT in the sample -- exactly _GATE_REPS runs happen and
    none is retried. (The median of [35.0, 35.1, 68.3] is still 35.1, which is why the returned
    value cannot tell these two modes apart; the retry count can.)
    """
    monkeypatch.setenv("PERF_MCP_THERMAL_GATE", "0")
    importlib.reload(mcp)
    calls = _feed(mcp, monkeypatch, [68.3, 35.0, 35.1], clamped={0})
    mcp._measure_full_pipeline_median()
    assert calls["n"] == mcp._GATE_REPS, calls


# ---------------------------------------------------------------- waiting for headroom


def test_it_waits_while_the_board_is_too_hot(mcp, monkeypatch):
    mcp._record_thermal_observation(78.3, clamped=True)
    temps = iter([85.0, 80.0, 68.0])
    monkeypatch.setattr(mcp, "_read_die_temp_c", lambda: next(temps, 68.0))
    ok, temp = mcp._wait_for_thermal_headroom()
    assert ok is True and temp == 68.0


def test_a_cool_board_does_not_wait(mcp, monkeypatch):
    mcp._record_thermal_observation(78.3, clamped=True)
    calls = {"n": 0}

    def temp():
        calls["n"] += 1
        return 65.0

    monkeypatch.setattr(mcp, "_read_die_temp_c", temp)
    mcp._wait_for_thermal_headroom()
    assert calls["n"] == 1, calls


def test_unreadable_telemetry_does_not_block_measuring(mcp, monkeypatch):
    """A board we cannot read the temperature of must not become a board we refuse to measure."""
    monkeypatch.setattr(mcp, "_read_die_temp_c", lambda: None)
    ok, temp = mcp._wait_for_thermal_headroom()
    assert ok is True and temp is None


def test_the_wait_gives_up_rather_than_hanging(mcp, monkeypatch):
    """A board that never cools must fall through to the clamp check, not stall the run forever."""
    monkeypatch.setenv("PERF_MCP_THERMAL_WAIT_S", "0")
    importlib.reload(mcp)
    mcp._record_thermal_observation(78.3, clamped=True)
    monkeypatch.setattr(mcp, "_read_die_temp_c", lambda: 85.0)
    ok, _temp = mcp._wait_for_thermal_headroom()
    assert ok is False


# ---------------------------------------------------------------- the threshold is LEARNED, not fixed


def test_an_unseen_board_does_not_wait_at_all(mcp, monkeypatch):
    """No hardcoded default. A fixed 70C would be wrong on any board that clamps below it, and
    would pass clamped readings through -- the exact failure this gate exists to stop. With no
    evidence the gate measures, and the clamp check teaches it."""
    assert mcp._clamp_threshold_c() is None
    calls = {"n": 0}
    monkeypatch.setattr(mcp, "_read_die_temp_c", lambda: (calls.__setitem__("n", calls["n"] + 1), 95.0)[1])
    ok, _t = mcp._wait_for_thermal_headroom()
    assert ok is True and calls["n"] == 1, calls


def test_one_clamp_establishes_a_threshold(mcp):
    """Backs off by the margin when only clamped starts are known."""
    mcp._record_thermal_observation(78.3, clamped=True)
    assert mcp._clamp_threshold_c() == round(78.3 - mcp._THERMAL_MARGIN_C, 2)


def test_the_threshold_lands_between_clean_and_clamped(mcp):
    """The real gemma3 observations: 69.8 and 73.5 held 1350; 78.3 and 79.9 clamped."""
    for t in (69.8, 73.5):
        mcp._record_thermal_observation(t, clamped=False)
    for t in (78.3, 79.9):
        mcp._record_thermal_observation(t, clamped=True)
    limit = mcp._clamp_threshold_c()
    assert 73.5 < limit < 78.3, limit


def test_a_cooler_clamp_lowers_the_threshold(mcp):
    """Different hardware clamps at a different point; the profile must follow the evidence down."""
    mcp._record_thermal_observation(78.3, clamped=True)
    first = mcp._clamp_threshold_c()
    mcp._record_thermal_observation(61.0, clamped=True)
    assert mcp._clamp_threshold_c() < first


def test_clean_starts_above_the_clamp_point_do_not_raise_it(mcp):
    """A lucky clean run at 80C must not license measuring at 80C -- clamping is probabilistic near
    the edge, and one success is not evidence of headroom."""
    mcp._record_thermal_observation(78.3, clamped=True)
    mcp._record_thermal_observation(80.0, clamped=False)
    assert mcp._clamp_threshold_c() <= 78.3


def test_the_env_override_still_wins(mcp, monkeypatch):
    mcp._record_thermal_observation(78.3, clamped=True)
    monkeypatch.setenv("PERF_MCP_MAX_START_TEMP_C", "55")
    assert mcp._clamp_threshold_c() == 55.0


def test_the_profile_survives_a_corrupt_file(mcp):
    mcp._thermal_profile_path().write_text("{not json")
    mcp._record_thermal_observation(78.3, clamped=True)
    assert mcp._clamp_threshold_c() is not None


def test_an_unknown_start_temperature_is_not_recorded(mcp):
    """Unreadable telemetry must not poison the profile with a fabricated number."""
    mcp._record_thermal_observation(None, clamped=True)
    assert mcp._clamp_threshold_c() is None


# ---------------------------------------------------------------- the clamp signal itself


def test_the_umd_clamp_warning_is_what_we_key_on(mcp):
    """UMD prints the clamp verbatim; keying on its own message beats sampling telemetry alongside,
    which aliases against a 17-second run."""
    line = (
        "AICLK failed to settle after 200 ms. Expected 1350, observed 800. ASIC temperature: "
        "78.28347778320312, AICLK clamped by max-arbiter index 7 at 800 MHz"
    )
    assert any(m in line for m in mcp._CLAMP_MARKERS)


def test_a_clean_run_does_not_look_clamped(mcp):
    assert not any(m in "TRACE_PER_TOKEN_MS=35.8279\n1 passed" for m in mcp._CLAMP_MARKERS)
