# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The path that measures every candidate must throw away a throttled reading, like the gate does.

IT ALREADY KNEW, AND USED THE NUMBER ANYWAY. probes.py has detected clock clamps since long before
the full-pipeline gate existed, and the profiling loop acted on it:

    if detect_overheat(log_text):
        _await_cool()

It cooled -- and then kept the reading. So it paid for the wait AND banked a number measured at
800 MHz instead of 1350. The wait only helped the NEXT run.

WHY THAT IS WORSE THAN CRASHING. These are the numbers every optimization decision is made from,
and a clamped one is a real timing of a slower machine:

    a good edit measured hot  -> looks ~40% slower -> reverted, never tried again
    a hot BASELINE            -> the next candidate looks faster -> a fake win is COMMITTED

Both corrupt the ledger silently. The full-pipeline gate has always discarded and retried; the
candidate path -- the one that actually picks what ships -- did not.

THE RULE, now the same in both places: a throttled run is not a measurement. Cool to an absolute
target, measure again, and if the board cannot hold its clock, fail loudly instead of returning a
number nothing can be compared against.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_SRC = (_PA / "agent" / "probes.py").read_text()


def _loop():
    """The profiling loop body, from the overheat check to the end of the retry block."""
    i = _SRC.index("if detect_overheat(log_text):")
    return _SRC[i : i + 2000]


def test_a_throttled_reading_is_discarded_not_kept():
    """THE BUG: cool, then use the bad number anyway."""
    body = _loop()
    assert "continue" in body, "a throttled run is still not re-measured"
    assert "DISCARDED" in body, "nothing records that the reading was thrown away"


def test_it_does_not_simply_cool_and_carry_on():
    """`_await_cool()` alone was the whole old behaviour, and it kept the reading."""
    i = _SRC.index("if detect_overheat(log_text):")
    j = _SRC.index("crash = detect_perf_crash", i)
    window = _SRC[i:j]
    assert "_await_cool()" not in window, "the cool-and-keep path is back"


def test_an_unmeasurable_board_raises_instead_of_returning_a_number():
    body = _loop()
    assert "raise ThrottledRunError" in body, "a permanently clamped board still yields a reading"


def test_the_error_is_not_mistaken_for_a_broken_edit():
    """PerfRunFailed routes to REPAIR_CODE and the agent starts rewriting correct code. A hot board
    is not a bug in the model, so this must not be that class."""
    assert "class ThrottledRunError(TracyRunError):" in _SRC
    i = _SRC.index("class ThrottledRunError")
    assert "PerfRunFailed" not in _SRC[i : i + 800]


def test_retries_are_bounded():
    """A board that clamps forever must terminate, not re-profile forever -- each attempt is minutes
    of device work plus a full model build."""
    assert "_MAX_THROTTLE_RETRIES" in _SRC
    body = _loop()
    assert "throttle_retry < _MAX_THROTTLE_RETRIES" in body


def test_the_counter_resets_per_measurement_not_per_process():
    """Left global, the second candidate of a run would inherit the first one's exhausted budget."""
    i = _SRC.index("heal_attempt = 0")
    assert "throttle_retry = 0" in _SRC[i : i + 120], "the throttle counter is not per-run state"


def test_it_cools_to_the_absolute_target_not_the_relative_courtesy_pause():
    """_await_cool is entry-5C capped at 120s. On a 96C board that asks for 91C -- still clamped."""
    i = _SRC.index("def _cool_before_remeasure(")
    body = _SRC[i : _SRC.index("\ndef ", i + 1)]
    assert "_cooldown_after_clamp" in body, "the re-measure cools by the relative rule"


def test_one_owner_for_cool_enough_to_measure():
    """Two definitions is how these two paths drifted apart in the first place."""
    i = _SRC.index("def _cool_before_remeasure(")
    body = _SRC[i : _SRC.index("\ndef ", i + 1)]
    # Delegation is the property; the IMPORT MECHANISM is not. A bare `from ..cc_optimize...` is
    # what this used to assert, and that import raises "attempted relative import beyond top-level
    # package" whenever probes is loaded as `agent.probes` -- which is how an optimize run loads it.
    # Inside `except: pass` that is silent, so on 2026-08-29 the board reached 99-102C and not one
    # thermal gate fired. _cc_optimize resolves the owner under either import shape.
    assert '_cc_optimize("perf_mcp")._cooldown_after_clamp()' in body, "does not delegate to the owner"
    assert "_await_cool()" in body, "no fallback if the import fails -- the run would die on a cooldown"
