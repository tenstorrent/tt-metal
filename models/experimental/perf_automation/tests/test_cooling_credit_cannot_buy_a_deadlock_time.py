# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Declared cooling pauses the watchdog clock. A claim of cooling must not pause it forever.

THE HOLE, caught by test_the_blind_backstop_stays_blind before it ever ran on hardware. Crediting
cooling time back was right -- a thermal wait is not work, and charging it to the op's budget is what
got a healthy run killed and the board wedged. The first implementation trusted a single BEGIN and
credited every second until END arrived. That gave away the one thing the absolute cap exists for.

A child that printed BEGIN and then busy-wait deadlocked would accrue credit exactly as fast as wall
clock, so `now - start - credit` would never grow and the cap could never fire -- while the stall
detector had also been told to treat cooling as liveness. Both guards off, forever, on one line of
output. Voxtral has produced that shape before: 85 minutes elapsed, 91 minutes of CPU, no output
after the first second.

THE RULE: credit is earned beat by beat, never extrapolated. The child re-asserts cooling every poll,
the parent banks the gap between consecutive beats, and a gap longer than _COOL_HEARTBEAT_S earns
nothing. A child that goes quiet stops being free immediately, whatever it last claimed.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_SRC = (_PA / "cc_optimize" / "run.py").read_text()


def _fn(name, end="def "):
    i = _SRC.index("def %s(" % name)
    return _SRC[i : _SRC.index(end, i + 10)]


def test_credit_is_banked_between_beats_not_extrapolated():
    body = _fn("_cool_beat")
    assert "now - prev <= _COOL_HEARTBEAT_S" in body, "an unbounded gap still earns credit"
    assert '_cool["total"] += now - prev' in body


def test_the_total_is_only_what_was_banked():
    """The bug in one line: returning banked + (now - since) makes an open claim self-renewing."""
    body = _fn("_cool_total")
    assert 'return _cool["total"]' in body
    assert "time.monotonic()" not in body, "the total still grows without a beat"


def test_a_silent_child_stops_counting_as_cooling():
    """The stall detector must not accept a stale claim as liveness either -- both guards were off."""
    body = _fn("_cooling_now")
    assert "_COOL_HEARTBEAT_S" in body, "cooling liveness is not time-bounded"
    i = _SRC.index("moved = _sig_moved")
    assert "_cooling_now()" in _SRC[i : i + 200], "the stall detector reads the raw flag again"


def test_end_clears_the_claim():
    i = _SRC.index("_COOL_END in _ln")
    assert '_cool["last"] = None' in _SRC[i : i + 200], "credit keeps accruing after the wait ended"


def test_the_child_re_asserts_every_poll():
    """Beats are the whole mechanism: a child that announces once and then sleeps quietly would
    have its credit cut off at the first heartbeat gap, and be killed mid-legitimate-cooldown."""
    child = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    for fn in ("_cooldown_after_clamp", "_headroom_poll"):
        i = child.index("def %s(" % fn)
        body = child[i : child.index("\ndef ", i + 1)]
        beats = body.count("_cooling_marker(_COOL_BEGIN)")
        assert beats >= 1, "%s never re-asserts" % fn
        j = body.index("time.sleep(")
        assert "_cooling_marker(_COOL_BEGIN)" in body[j : j + 400], "%s does not beat inside its loop" % fn


def test_the_heartbeat_window_is_wider_than_the_poll():
    """20 s polls against a 90 s window: two beats can be lost to a slow telemetry read without the
    wait being mistaken for a hang."""
    import re

    m = re.search(r'_COOL_HEARTBEAT_S = float\(os\.environ\.get\("PERF_MCP_COOL_HEARTBEAT_S", "(\d+)"\)\)', _SRC)
    assert m, "the window is not configurable"
    child = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    poll = re.search(r'_COOLDOWN_POLL_S = float\(os\.environ\.get\("PERF_MCP_COOLDOWN_POLL_S", "(\d+)"\)\)', child)
    assert poll and float(m.group(1)) >= 3 * float(poll.group(1))
