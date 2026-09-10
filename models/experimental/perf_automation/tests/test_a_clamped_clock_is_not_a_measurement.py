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
    # The state dir must be a SUBdirectory: the thermal profile is a board fact and now lives one
    # level up, so pointing the state dir straight at tmp_path would put it in pytest's shared root
    # and let each test inherit the clamp observations recorded by the last one.
    _sd = tmp_path / "model"
    _sd.mkdir(exist_ok=True)
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(_sd))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(_sd))
    monkeypatch.delenv("PERF_MCP_THERMAL_GATE", raising=False)
    # The suite disables the start gate for every other test (conftest, PERF_MCP_MAX_START_TEMP_C).
    # These tests ARE the gate, so they take the stated threshold and drive a mocked thermometer.
    monkeypatch.delenv("PERF_MCP_MAX_START_TEMP_C", raising=False)
    monkeypatch.setenv("PERF_MCP_THERMAL_POLL_S", "0")
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _feed(mcp, monkeypatch, readings, clamped=(), cooled=True):
    """Each reading is one run; `clamped` marks which run indices report a clamped clock.

    The post-clamp cooldown is stubbed here because these tests are about WHICH reading is kept, not
    about the wait. Left real, they read the actual die temperature and sleep against it -- on a busy
    board that is a 30-minute test, which is how this stub came to be needed.
    """
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
    monkeypatch.setattr(mcp, "_cooldown_after_clamp", lambda *_a, **_k: (bool(cooled), 55.0))
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


def test_a_board_that_will_not_cool_stops_after_the_first_clamp(mcp, monkeypatch):
    """THE RUN-13 CASE. Four attempts, an hour, every reading discarded, board 79C -> 96C. If the
    board cannot get back to the cooldown target there is nothing to be gained by measuring again:
    stop at the first clamp and say why, instead of spending three more runs proving it."""
    calls = _feed(mcp, monkeypatch, [68.3] * 12, clamped=set(range(12)), cooled=False)
    ms, _method, err, _path = mcp._measure_full_pipeline_guarded()
    assert ms is None
    assert calls["n"] == 1, "kept measuring a board that never reached the cooldown target: %s" % calls
    assert "cool" in (err or "").lower(), err


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
    temps = iter([85.0, 80.0, 58.0])
    monkeypatch.setattr(mcp, "_read_die_temp_c", lambda: next(temps, 58.0))
    ok, temp = mcp._wait_for_thermal_headroom()
    assert ok is True and temp == 58.0


def test_a_cool_board_does_not_wait(mcp, monkeypatch):
    calls = {"n": 0}

    def temp():
        calls["n"] += 1
        return 55.0

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


# ---------------------------------------------------------------- the threshold is STATED, not learned
#
# IT WAS LEARNED, AND ON 2026-08-16 THE LEARNING PUT IT BELOW THE BOARD'S IDLE TEMPERATURE. The rule
# was min(clamped_at) - 3, where clamped_at holds the temperature a run STARTED at, written down
# whenever that run clamped at some later point. A run that began at 56.75C, heated for twenty
# minutes and clamped at 85C therefore recorded 56.75 -- and min() handed that one sample, out of
# 39, authority over all of them:
#
#     min(clamped) - 3  =  53.8C        this board idles at 53.9C
#
# The gate could never pass. It waited its full 900s at every measurement and then measured hot,
# which is the outcome it exists to prevent, reached by way of a quarter-hour delay. And min() only
# moves down, so no later evidence could undo it.
#
# THE SIGNAL DOES NOT PREDICT. Across 177 recorded runs the clamped starts (n=39) have median 72.5C
# and the clean ones (n=138) median 70.8C, on ranges that almost entirely overlap. Starting below
# 68C moves the clamp rate from 22% to 10%; starting below 60C measured WORSE than average. There is
# no threshold through those two distributions, so no statistic computed from them was going to
# work -- which is why this is stated rather than fixed with a better formula.


def test_the_threshold_is_the_stated_number(mcp):
    assert mcp._clamp_threshold_c() == 65.0


def test_it_sits_above_the_temperature_the_cooldown_holds_to(mcp):
    """So a board that has just completed a post-clamp cooldown is always clear to start again. The
    other way round -- a gate stricter than the cooldown -- would cool to 60C, be refused, and wait
    on a board already as cold as the tool knows how to make it."""
    assert mcp._clamp_threshold_c() > mcp._COOLDOWN_TO_C


def test_evidence_does_not_move_it(mcp):
    """THE RATCHET, GONE. This is the regression: one cold-start-then-clamp sample used to drag the
    threshold below the board's idle temperature, permanently, because min() only moves down."""
    assert mcp._clamp_threshold_c() == 65.0
    mcp._record_thermal_observation(56.75, clamped=True)
    mcp._record_thermal_observation(87.2, clamped=True)
    mcp._record_thermal_observation(70.5, clamped=False)
    assert mcp._clamp_threshold_c() == 65.0


def test_nothing_in_the_threshold_reads_the_profile(mcp):
    """Any rule that consults the samples is a rule the samples can move, which is the whole defect.
    Asserted on the code, because a value test cannot tell a constant from a statistic that happens
    to agree with it today."""
    import inspect

    src = inspect.getsource(mcp._clamp_threshold_c)
    body = src.split('"""', 2)[-1]
    for name in ("clamped_at", "clean_at", "_load_thermal_profile", "min(", "sorted("):
        assert name not in body, "the threshold reads the profile again via %s" % name


def test_an_unseen_board_uses_the_stated_number(mcp, monkeypatch):
    """No history is not a reason to skip the gate. The old bootstrap returned None -- 'measure, and
    the clamp check will teach us' -- which was the right answer only while the number was learned."""
    calls = {"n": 0}
    monkeypatch.setattr(mcp, "_read_die_temp_c", lambda: (calls.__setitem__("n", calls["n"] + 1), 55.0)[1])
    ok, _t = mcp._wait_for_thermal_headroom()
    assert ok is True and calls["n"] == 1, calls


def test_the_env_override_still_wins(mcp, monkeypatch):
    mcp._record_thermal_observation(78.3, clamped=True)
    monkeypatch.setenv("PERF_MCP_MAX_START_TEMP_C", "55")
    assert mcp._clamp_threshold_c() == 55.0


def test_the_profile_survives_a_corrupt_file(mcp):
    mcp._thermal_profile_path().write_text("{not json")
    mcp._record_thermal_observation(78.3, clamped=True)
    assert mcp._clamp_threshold_c() is not None


def test_an_unknown_start_temperature_is_not_recorded(mcp):
    """Unreadable telemetry must not poison the profile with a fabricated number.

    Asserted on the profile itself. It used to be asserted through the threshold coming back None,
    which stopped meaning anything once the threshold was stated -- the test would have passed on a
    profile full of invented readings."""
    mcp._record_thermal_observation(None, clamped=True)
    doc = mcp._load_thermal_profile()
    assert not (doc.get("clamped_at") or doc.get("clean_at")), doc


# ---------------------------------------------------------------- mesh scoping


def _smi(mcp, monkeypatch, temps):
    """Drive the TT-SMI parsing path with a scripted payload.

    sysfs is stubbed empty because it is now tried FIRST, and on a host that has hwmon it answers in
    0.3 ms -- the real board's temperature, not this payload. These tests are about the tt-smi
    fallback, so the faster source has to be taken away for them to reach it.
    """
    payload = {"device_info": [{"telemetry": {"asic_temperature": t}} for t in temps]}

    class R:
        stdout = __import__("json").dumps(payload)

    from agent import probes as _pr

    monkeypatch.setattr(_pr, "_sysfs_asic_temps", lambda: [])
    monkeypatch.setattr(_pr.subprocess, "run", lambda *a, **k: R())
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

    from agent import probes as _pr

    monkeypatch.setattr(_pr, "_sysfs_asic_temps", lambda: [])  # sysfs answers first; take it away
    monkeypatch.setattr(_pr.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(mcp._sp, "run", lambda *a, **k: R())
    assert mcp._read_die_temp_c() is None


def test_a_chip_with_no_temperature_does_not_break_the_read(mcp, monkeypatch):
    payload = {"device_info": [{"telemetry": {}}, {"telemetry": {"asic_temperature": 71.0}}]}

    class R:
        stdout = __import__("json").dumps(payload)

    from agent import probes as _pr

    monkeypatch.setattr(_pr, "_sysfs_asic_temps", lambda: [])  # sysfs answers first; take it away
    monkeypatch.setattr(_pr.subprocess, "run", lambda *a, **k: R())
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
