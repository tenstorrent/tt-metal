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
    body = src[i : i + 9000]
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


def test_the_blind_backstop_stays_blind():
    """Deliberate. The stall detector already checks CPU, output and live children; the absolute cap
    exists precisely for the case where those are fooled -- a busy-wait deadlock. Voxtral produced
    one: 85 minutes, 91 minutes of CPU, no log output after the first second."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("if now - start >= timeout_s:")
    stanza = src[i : i + 200]
    assert "moved" not in stanza and "cpu" not in stanza, "the absolute cap now consults activity"
