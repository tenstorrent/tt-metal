# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Waiting for a board to cool is not work, and it is not a hang.

WHAT BROKE THE BOARD ON 2026-08-14. A thermal wait runs INSIDE the device subprocess, and the parent
watchdog times that subprocess on wall clock. So the tool cooled the board, the cooling consumed the
op's budget, and the watchdog killed it:

    termination_check KILLED after 1716s (hard limit) (likely a device wedge / leaked mesh)
      -- killed the whole process group + reclaimed device

Nothing checked the temperature at kill time; "likely a device wedge" is printed on every timeout.
The board was not wedged. The RESET that the kill triggered is what wedged it -- device 2 dropped
into `Failed to set initial power state: -22`, which no PCIe reset clears, and the host needed a
reboot. The tool cooled the board, killed itself for taking too long, and then broke the hardware
diagnosing a wedge that did not exist.

TWO RULES, both tested here.

    1. Cooling time is credited back, so an hour of cooling costs an hour and none of the budget.
    2. Cooling is not a stall. A cooling child sleeps against a thermometer: no CPU, and no output
       between polls. Both liveness signals read that as a wedge, which is exactly backwards.

And on the cooling side: there is no timer on physics. A board cools at the rate it cools, and a
deadline can only cut the wait short and hand back a hot board. The wait ends on EVIDENCE -- either
the target is reached, or the temperature has stopped falling, which means the board has found its
floor (this chassis idles at 79C) and no further waiting will change it.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


# ------------------------------------------------------------------ the child brackets its waits


def test_both_thermal_waits_emit_the_markers():
    """A wait the parent cannot see is a wait the parent kills."""
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    for fn in ("_cooldown_after_clamp", "_wait_for_thermal_headroom"):
        i = src.index("def %s(" % fn)
        body = src[i : src.index("\ndef ", i + 1)]
        assert "_cooling_marker(_COOL_BEGIN)" in body, "%s does not announce that it is cooling" % fn
        assert "_COOL_END" in body, "%s does not announce that it stopped" % fn
        assert "finally:" in body, "%s leaves the clock stopped if it raises" % fn


def test_the_parent_and_child_agree_on_the_marker_text():
    child = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    parent = (_PA / "cc_optimize" / "run.py").read_text()
    for name in ('_COOL_BEGIN = "PERF_MCP_COOLING_BEGIN"', '_COOL_END = "PERF_MCP_COOLING_END"'):
        assert name in child and name in parent, "the two ends disagree about %r" % name


# ------------------------------------------------------------------ the parent stops its clock


def test_the_hard_limit_excludes_cooling():
    """Anchored on the PROPERTY, not the spelling. The comparison was once written inline and is now
    hoisted into `_worked`; either is fine, so long as cooling is subtracted before the budget and
    the ceiling are judged."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    assert "_worked = now - start - _cool_total()" in src, "the work clock no longer discounts cooling"
    k = src.index("_worked = now - start - _cool_total()")
    stanza = src[k : k + 1800]
    assert "_worked >= timeout_s" in stanza, "the budget is not measured on the cooling-discounted clock"
    assert "_worked >= timeout_s * _ceiling_mult" in stanza, "the ceiling is not measured on it either"


def test_a_cooling_child_is_not_a_stall():
    """It burns no CPU and prints only when the temperature moves -- both wedge signals, wrongly."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    # CPU is no longer a liveness signal at all (a livelock has plenty); the line is now
    # `moved = _sig_moved or _act[0] > last_progress or _cooling_now()`. What this test is about is
    # that cooling still counts, and it still does.
    i = src.index("moved = _sig_moved")
    assert "_cooling_now()" in src[i : i + 200], "a cooling child still reads as a no-progress stall"


def test_an_unfinished_cooldown_is_credited_beat_by_beat_not_extrapolated():
    """This test used to assert the opposite, and the opposite was the bug.

    The worry was real -- a kill landing mid-cooldown must not lose the credit for the wait in
    progress -- but crediting `now - since` for an open claim meant a child could announce a cooldown,
    deadlock, and accrue credit as fast as wall clock, so the absolute cap could never fire. The
    in-progress wait is still credited; it is credited by the beats already banked, which stop the
    moment the child does. See test_cooling_credit_cannot_buy_a_deadlock_time.py."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _cool_beat()")
    beat = src[i : src.index("def _cool_total()", i)]
    assert '_cool["total"] += now - prev' in beat, "an in-progress wait banks nothing until it ends"
    j = src.index("def _cool_total()")
    assert "time.monotonic()" not in src[j : j + 120], "the total is extrapolated past the last beat"


# ------------------------------------------------------------------ no timer on physics


def _mcp(monkeypatch, temps, target=60.0):
    from cc_optimize import perf_mcp

    seen = iter(temps)
    monkeypatch.setattr(perf_mcp, "_read_die_temp_c", lambda: next(seen, temps[-1]))
    monkeypatch.setattr(perf_mcp.time, "sleep", lambda _s: None)
    monkeypatch.setattr(perf_mcp, "_COOLDOWN_TO_C", target)
    monkeypatch.setattr(perf_mcp, "_COOLDOWN_POLL_S", 0.0)
    return perf_mcp


def test_a_slow_cooler_is_waited_out_however_long_it_takes(monkeypatch):
    """THE POINT. 200 polls of slow progress is a slow board, not a broken one -- there is no
    deadline that could tell those apart, so there is no deadline."""
    mcp = _mcp(monkeypatch, [95.0 - 0.2 * i for i in range(200)])
    ok, reached = mcp._cooldown_after_clamp()
    assert ok and reached <= 60.0, reached


def test_a_board_that_stops_falling_is_reported_not_waited_on_forever(monkeypatch):
    """79C is this chassis's floor. Waiting past that is waiting for something that will not happen."""
    mcp = _mcp(monkeypatch, [95.0, 88.0, 82.0, 79.0] + [79.0] * 50)
    monkeypatch.setattr(mcp, "_COOLDOWN_PLATEAU_S", 0.0)
    ok, reached = mcp._cooldown_after_clamp()
    assert not ok and reached == 79.0, (ok, reached)


def test_the_plateau_is_measured_from_the_last_improvement_not_from_the_start(monkeypatch):
    """A board still inching down must never trip the plateau rule."""
    mcp = _mcp(monkeypatch, [95.0, 90.0, 85.0, 80.0, 70.0, 59.0])
    monkeypatch.setattr(mcp, "_COOLDOWN_PLATEAU_S", 0.0)  # would fire instantly on any flat poll
    ok, reached = mcp._cooldown_after_clamp()
    assert ok and reached <= 60.0, "steady progress was mistaken for a plateau"


def test_no_wall_clock_deadline_survives_in_the_cooldown():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("def _cooldown_after_clamp(")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "_COOLDOWN_MAX_S" not in body, "a timer on physics is back"
    assert "_COOLDOWN_PLATEAU_S" in body, "nothing decides when to give up"
