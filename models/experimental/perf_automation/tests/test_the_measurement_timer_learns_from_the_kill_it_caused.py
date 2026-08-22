# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The profile budget is derived from observed profile durations, so profiles must record them.

WHAT THIS COST, measured on Voxtral-Mini-3B, 2026-08-11.

`_measure_backstop` documents itself as "hard wall for one on-device measurement, derived from
observed PROFILE durations". Nothing ever recorded one. `record_observed` had exactly two callers,
"pcc" and "round", so `_op_cost("profile")` was 0 forever, `adaptive_timer` took its `cost <= 0`
cold-start branch on every call, and every measurement on every model got the same fixed guess.

The baseline profile was killed at 900 s -- the "hard limit", not the stall detector -- while still
printing `decode_trace_step #96`, i.e. while demonstrably working. The run then optimized for hours
with no BEFORE number, because a failed baseline is laundered into "manifest is complete;
continuing". A retry would have used the same 900 s, because a kill taught the timer nothing.

That is the part worth fixing. A hard-limit kill is the STRONGEST available evidence that a budget
was too small, and it was the one outcome the timer never saw. Recording only successes means the
timer can only learn from runs that did not need it to.

Note what is deliberately NOT changed: the blind backstop stays blind. Making it activity-aware
would recreate the orphan this same run produced -- a process spinning in a poll loop for 85 minutes
looks identical to hard work from outside, and only an unconditional cap ever stops it. The fix is
the backstop's VALUE, not its existence.
"""

from pathlib import Path

_PA = Path(__file__).resolve().parent.parent


def _run():
    from models.experimental.perf_automation.cc_optimize import run as _r

    return _r


def test_the_profile_op_records_its_duration():
    """Without this the 'profile' series is always empty and the budget is always the cold start."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    assert src.count('observe_op="profile"') >= 3, "profile call sites do not record their cost"


def test_it_records_on_the_timeout_path_too():
    """A kill is evidence the budget was too small. Recording only on success discards exactly the
    measurement that would have corrected it."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _run_device_proc(")
    # Scoped to the function, not a fixed character count: a 9000-char window silently stopped
    # covering the recording as soon as the timeout handler grew.
    body = src[i : src.index("\ndef ", i + 1)]
    rec = body.index("record_observed(observe_root, observe_op")
    fin = body.rindex("finally:", 0, rec)
    # the recording must sit inside a `finally`, so SIGKILL/TimeoutExpired still reaches it
    assert fin < rec, "the duration is recorded outside finally, so a killed run records nothing"


def test_the_backstop_still_derives_from_that_series():
    """The recording is only useful because the wall is computed from it; if the wall stopped
    reading 'profile', the recording would be dead weight."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _measure_backstop(")
    assert '"profile"' in src[i : i + 400], "the hard wall no longer derives from profile cost"


def test_the_backstop_no_longer_kills_code_that_is_working():
    """REVERSED ON 2026-08-17, deliberately, and the case it was written for is still real.

    It used to assert the absolute cap must stay BLIND to activity: "the stall detector already
    checks CPU, output and live children; the absolute cap exists precisely for the case where those
    are fooled -- a busy-wait deadlock. Voxtral produced one: 85 minutes, 91 minutes of CPU, no log
    output after the first second."

    That reasoning stands. What changed is the price of being wrong in the other direction. On
    2026-08-17 the cap killed a full-depth measurement at three hours while its tree was burning CPU
    and the board sat at 97-102C -- work in progress, ended on a number that was not a judgement
    about this measurement at all: it was the CEILING, taken only because --fresh had wiped the
    observed durations meant to size it. And the kill does not merely discard the work; it triggers
    the recovery, and that recovery reset four chips to rescue a one-chip run and left two of them
    dead. Twice in one day.

    So the rule is now the one the cooldown already follows: a run ends on evidence of death, not on
    elapsed time. The stall clock still kills the moment CPU, output, live children and cooling all
    go quiet -- which is what a wedge looks like.

    THE COST, ACCEPTED KNOWINGLY: a busy-wait deadlock -- CPU moving, nothing produced -- now runs
    until it stops or an operator stops it. The 85-minute voxtral deadlock would today run
    indefinitely. That is the trade: this tool would rather spend hours on work that may finish than
    destroy work that was finishing, because destroying it also costs chips.

    If the trade ever needs revisiting, the fix is not to restore a blind clock -- it is to make
    OUTPUT the liveness signal instead of CPU, since that deadlock printed nothing after one second
    and would have been caught in minutes.

    UPDATED 2026-08-20, BY THE CASE THIS DOCSTRING PREDICTED. It ends "the fix is not to restore a
    blind clock -- it is to make OUTPUT the liveness signal instead of CPU". Run 12 is what it cost
    to leave that undone: at 03:09, exactly 3h in, the loop printed "over its 10800s budget and
    STILL WORKING (tree CPU is moving) -- not killing it", then produced NOT ONE BYTE for nine hours
    while holding the board, until it was killed by hand.

    So liveness is now a progress signature -- log bytes, syscalls, io bytes, stack movement -- and
    CPU is not consulted anywhere. The trade this docstring accepted ("a busy-wait deadlock now runs
    until an operator stops it") is no longer accepted, because the operator turned out to be the
    only backstop and cost nine hours and a wedged board.

    What is preserved: the budget itself still does NOT kill. Reaching 1x the budget is reported,
    once, and the run continues -- that part was right, and killing on it destroyed real work twice.
    What is added: a ceiling at _HARD_CEILING_MULT x the budget that does kill, because a signature
    that keeps twitching without ever finishing is not work either.
    """
    src = (_PA / "cc_optimize" / "run.py").read_text()
    k = src.index("if not _over_budget[0] and _worked >= timeout_s:")
    stanza = src[k : k + 700]
    assert "TimeoutExpired" not in stanza, "the budget itself kills again instead of reporting"
    assert "STILL WORKING" in stanza, "the over-budget case is no longer announced"
    assert "tree CPU is moving" not in src, "the message still points at CPU, which is not the signal"
    assert "raise subprocess.TimeoutExpired(cmd, limit)" in src, "nothing kills a genuinely stalled run"

    # ...and the ceiling behind it does kill.
    c = src.index("if _ceiling_mult and timeout_s and _worked >= timeout_s * _ceiling_mult:")
    assert "raise subprocess.TimeoutExpired" in src[c : c + 900], "the ceiling does not stop anything"
