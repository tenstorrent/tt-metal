# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""After the hardware throttles, the next attempt starts from a COLD board or it does not start.

WHAT WENT WRONG. The measure loop discards a reading whose AICLK was clamped -- correctly, a number
taken at 800 MHz is not comparable to one taken at 1350 -- and then retries. The retry called
_wait_for_thermal_headroom, which is bounded at 900 s and RUNS ANYWAY when the board has not cooled.

On Voxtral-Mini-3B, 2026-08-14, that produced a loop that could not converge: each attempt waited its
900 s, gave up, measured hot, was discarded for clamping, and left the board hotter than it found it.
The board went 79C -> 96C over the run. Four attempts, an hour of device time, nothing measured:

    status=crash end_to_end_ms=None
    error=every reading was discarded: AICLK was clamped on all 4 attempts

THE RULE. A clamped reading is evidence the board is too hot to measure, so the retry waits for an
ABSOLUTE target (60C by default) and does not abandon that wait on a timer the way the pre-run
headroom check does. A relative target is no use here -- "entry minus 5" on a 96C board asks for 91C,
which is still clamped. If the board never reaches the target the run says so, which is a far more
useful outcome than four more readings at 800 MHz.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _mcp(monkeypatch, temps, target=60.0):
    """perf_mcp with a scripted temperature trace and no real sleeping."""
    from cc_optimize import perf_mcp

    seen = iter(temps)
    monkeypatch.setattr(perf_mcp, "_read_die_temp_c", lambda: next(seen, temps[-1]))
    monkeypatch.setattr(perf_mcp.time, "sleep", lambda _s: None)
    monkeypatch.setattr(perf_mcp, "_COOLDOWN_TO_C", target)
    monkeypatch.setattr(perf_mcp, "_COOLDOWN_POLL_S", 0.0)
    return perf_mcp


def test_it_holds_until_the_absolute_target_not_a_relative_one(monkeypatch):
    """96C entry must not be satisfied by 91C. THE BUG _await_cool would have had here."""
    mcp = _mcp(monkeypatch, [96.0, 91.0, 84.0, 70.0, 59.5])
    ok, reached = mcp._cooldown_after_clamp()
    assert ok
    assert reached <= 60.0, "released the retry at %.1fC, above the 60C target" % reached


def test_a_board_already_cold_is_not_delayed(monkeypatch):
    mcp = _mcp(monkeypatch, [42.0])
    ok, reached = mcp._cooldown_after_clamp()
    assert ok and reached == 42.0


def test_a_board_that_never_cools_stops_the_run_instead_of_retrying_hot(monkeypatch):
    """This board idles at 79C. Once it stops falling, 60C is unreachable and the tool must SAY SO
    rather than measure at 800 MHz. No timer decides this -- the temperature not moving does."""
    mcp = _mcp(monkeypatch, [95.0, 90.0, 82.0, 79.0, 79.0, 79.0])
    monkeypatch.setattr(mcp, "_COOLDOWN_PLATEAU_S", 0.0)  # it has stopped falling; that is the signal
    monkeypatch.setattr(mcp.time, "sleep", lambda _s: None)
    ok, reached = mcp._cooldown_after_clamp()
    assert not ok, "gave the retry a green light from a board that never reached the target"
    assert reached is not None


def test_unreadable_telemetry_does_not_become_a_board_we_refuse_to_use(monkeypatch):
    """Same rule the rest of the thermal path follows: a missing sensor is not a hot board."""
    mcp = _mcp(monkeypatch, [95.0, None])
    ok, _reached = mcp._cooldown_after_clamp()
    assert ok


def test_the_discard_path_actually_calls_it():
    """The cooldown is worthless if the retry loop still goes straight back in."""
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("DISCARDED %.4f ms")
    window = src[i : src.index("return float(ms)", i)]
    assert "_cooldown_after_clamp()" in window, "a discarded reading still retries without cooling"
    assert "return (" in window, "an unreachable target still falls through into another hot attempt"


def test_the_target_is_absolute_and_the_wait_is_not_the_bounded_headroom_one():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("def _cooldown_after_clamp(")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "_COOLDOWN_TO_C" in body
    assert "_wait_for_thermal_headroom(" not in body, "it delegates to the wait that gives up"
