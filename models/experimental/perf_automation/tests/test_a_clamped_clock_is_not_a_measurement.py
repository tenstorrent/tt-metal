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
    ms, *_ = mcp._measure_full_pipeline_guarded()
    assert ms is not None and ms < 36, ms


def test_a_discarded_reading_is_replaced_not_dropped(mcp, monkeypatch):
    """Two clamped readings then a clean one: three runs, and the clean one is the answer."""
    calls = _feed(mcp, monkeypatch, [68.3, 69.9, 35.0], clamped={0, 1})
    ms, *_ = mcp._measure_full_pipeline_guarded()
    assert calls["n"] == 3 and ms == 35.0, (calls, ms)


def test_an_all_clamped_board_fails_loudly(mcp, monkeypatch):
    """No number at all beats a wrong number: this is what stops a bad BEFORE anchor being written."""
    _feed(mcp, monkeypatch, [68.3] * 12, clamped=set(range(12)))
    ms, _method, err, _path = mcp._measure_full_pipeline_guarded()
    assert ms is None and "clamped" in (err or "").lower(), (ms, err)


def test_retries_are_bounded(mcp, monkeypatch):
    """A permanently hot board must terminate, not spin forever."""
    calls = _feed(mcp, monkeypatch, [68.3] * 50, clamped=set(range(50)))
    mcp._measure_full_pipeline_guarded()
    assert calls["n"] <= 1 + mcp._THERMAL_RETRIES, calls


def test_a_clean_reading_is_taken_once(mcp, monkeypatch):
    """ONE reading. Repeating it was tried and removed: a 17s measurement is itself what heats the
    board past the clamp point, so extra reps manufacture the condition that invalidates them --
    on gemma3 the median of three WAS the clamped 68.32 against a true 35."""
    calls = _feed(mcp, monkeypatch, [35.0, 35.1, 35.2])
    ms, _m, err, _p = mcp._measure_full_pipeline_guarded()
    assert err is None and ms == 35.0 and calls["n"] == 1, (ms, err, calls)


def test_the_gate_can_be_switched_off(mcp, monkeypatch):
    """An escape hatch for a board whose telemetry lies, or a bring-up on unknown silicon.

    Switched off, the clamped reading is KEPT: one run happens and nothing is retried.
    """
    monkeypatch.setenv("PERF_MCP_THERMAL_GATE", "0")
    importlib.reload(mcp)
    calls = _feed(mcp, monkeypatch, [68.3, 35.0, 35.1], clamped={0})
    mcp._measure_full_pipeline_guarded()
    assert calls["n"] == 1, calls


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


# ---------------------------------------------------------------- mesh scoping


def _smi(mcp, monkeypatch, temps):
    payload = {"device_info": [{"telemetry": {"asic_temperature": t}} for t in temps]}

    class R:
        stdout = __import__("json").dumps(payload)

    monkeypatch.setattr(mcp._sp, "run", lambda *a, **k: R())


def test_it_takes_the_max_across_the_mesh(mcp, monkeypatch):
    """A collective runs at the pace of its slowest chip, so one hot member governs.

    Scoping to the mesh needs no filtering here: tt-smi honours TT_VISIBLE_DEVICES itself, so what
    it returns IS the mesh. Verified on a 4-chip p300c -- unset reports 4 chips, "0" reports 1,
    "0,1" and "2,3" each report 2. Filtering on top would break "2,3", which tt-smi returns at list
    positions 0 and 1.
    """
    _smi(mcp, monkeypatch, [60.0, 85.0])
    assert mcp._read_die_temp_c() == 85.0


def test_a_single_chip_reading_is_that_chip(mcp, monkeypatch):
    _smi(mcp, monkeypatch, [57.1])
    assert mcp._read_die_temp_c() == 57.1


def test_unparseable_telemetry_reads_as_unknown(mcp, monkeypatch):
    """What a stale TT_VISIBLE_DEVICES actually produces: tt-smi exits non-zero with no JSON.
    Unknown must mean proceed, not refuse."""

    class R:
        stdout = "Error: device 99 not found\n"

    monkeypatch.setattr(mcp._sp, "run", lambda *a, **k: R())
    assert mcp._read_die_temp_c() is None


def test_a_chip_with_no_temperature_does_not_break_the_read(mcp, monkeypatch):
    payload = {"device_info": [{"telemetry": {}}, {"telemetry": {"asic_temperature": 71.0}}]}

    class R:
        stdout = __import__("json").dumps(payload)

    monkeypatch.setattr(mcp._sp, "run", lambda *a, **k: R())
    assert mcp._read_die_temp_c() == 71.0


# ---------------------------------------------------------------- the clamp signal itself


def test_the_umd_clamp_warning_is_what_we_key_on(mcp):
    """UMD prints the clamp verbatim; keying on its own message beats sampling telemetry alongside,
    which aliases against a 17-second run."""
    line = (
        "AICLK failed to settle after 200 ms. Expected 1350, observed 800. ASIC temperature: "
        "78.28347778320312, AICLK clamped by max-arbiter index 7 at 800 MHz"
    )
    assert mcp._run_reported_clamp(line)


def test_a_clean_run_does_not_look_clamped(mcp):
    assert not mcp._run_reported_clamp("TRACE_PER_TOKEN_MS=35.8279\n1 passed")


def test_it_reuses_the_existing_overheat_detector(mcp):
    """The tool already had detect_overheat guarding the tracy path. A second detector would be one
    more thing to keep in step; this asserts the gate goes through the shared one."""
    from models.experimental.perf_automation.agent.probes import detect_overheat

    line = "AICLK failed to settle after 200 ms. Expected 1350, observed 800."
    assert detect_overheat(line), "probes must recognise the CURRENT UMD wording"
    assert mcp._run_reported_clamp(line)


def test_a_clamp_without_the_arbiter_tag_is_still_caught(mcp):
    """AICLK_ARB_MAX is guarded by is_entry_available(), so on a part whose telemetry enum lacks it
    UMD prints only the settle failure. Keying solely on 'AICLK clamped' missed exactly that."""
    assert mcp._run_reported_clamp("AICLK failed to settle after 200 ms. Expected 1000, observed 500.")
