# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A step that was KILLED must not be reported as a step that FAILED.

A step reports through two channels: its OUTPUT and its EXIT STATUS. A signal death says nothing in
the first -- SIGKILL has no handler, so there is no traceback and the output stops mid-line -- and
everything downstream reads the output. `rc = -9` was therefore routed to the same branch as
`rc = 1`, and "the tool killed this step" reached the agent as "some unspecified invalid result",
with a directive telling it to fix code that was never the problem.

On a Qwen-Image-Edit bring-up five captures died with rc=-9, no traceback and no watchdog line, over
two days. Nothing recorded who sent the signal, and nothing distinguished a kill from a failure.
Two things fix that: the exit status is now read (a negative returncode IS the signal), and the
killer announces itself, since the victim provably cannot.
"""

from __future__ import annotations

import os
import subprocess
import sys

from models.experimental.perf_automation.agent import probes as PR
from models.experimental.perf_automation.agent.perf_test_gen import signal_note

# --- the exit-status channel --------------------------------------------------------------------


def test_a_signal_death_is_named():
    assert signal_note(-9) == "terminated by SIGKILL (rc=-9)"
    assert signal_note(-15) == "terminated by SIGTERM (rc=-15)"
    assert signal_note(-6) == "terminated by SIGABRT (rc=-6)"


def test_an_ordinary_exit_says_nothing():
    """A failing test is not a killed test -- rc=1 and rc=124 must stay as they were."""
    for rc in (0, 1, 2, 124):
        assert signal_note(rc) == ""


def test_a_missing_or_junk_rc_does_not_raise():
    for rc in (None, "", "x", object()):
        assert signal_note(rc) == ""


def test_an_unknown_signal_number_still_reports():
    got = signal_note(-99)
    assert "signal 99" in got and "rc=-99" in got


def test_the_verdict_leads_with_the_signal(monkeypatch):
    """What the gate reads must say "killed", not just "invalid"."""
    import models.experimental.perf_automation.agent.perf_test_gen as PG

    monkeypatch.setattr(PG, "_run_perf_node", lambda *a, **k: (-9, "some truncated output\n"))
    monkeypatch.setattr(PG, "_write_trace_caps", lambda *a, **k: None)
    status, detail = PG.validate_generated_perf_test(__file__, "main")
    assert status == "invalid"
    assert detail.startswith("terminated by SIGKILL")
    assert "killed, not failed" in detail


def test_an_ordinary_failure_is_unchanged(monkeypatch):
    import models.experimental.perf_automation.agent.perf_test_gen as PG

    monkeypatch.setattr(PG, "_run_perf_node", lambda *a, **k: (1, "E   AssertionError: pcc too low\n"))
    monkeypatch.setattr(PG, "_write_trace_caps", lambda *a, **k: None)
    status, detail = PG.validate_generated_perf_test(__file__, "main")
    assert status == "invalid"
    assert "terminated by" not in detail, "a failure must not be described as a kill"


# --- the killer announcing itself ---------------------------------------------------------------


def test_the_killer_names_itself_and_its_victim(capfd):
    """The victim cannot report a SIGKILL, so the sender must. Uses a real short-lived child."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        PR._kill_tree(child.pid)
    finally:
        child.wait(timeout=10)
    err = capfd.readouterr().err
    assert "[kill] SIGKILL" in err
    assert "pid %d" % os.getpid() in err, "the killer must identify itself"
    assert "pid %d" % child.pid in err, "and name what it killed"


def test_the_signature_never_raises_on_a_dead_pid():
    """It reads /proc, which races with the process dying."""
    assert "pid" in PR._kill_signature(2**22 - 1)


def test_the_announcement_goes_to_stderr_not_the_captured_log(capfd):
    """It must not land inside the step's own log, where it would look like the step said it."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        PR._kill_tree(child.pid)
    finally:
        child.wait(timeout=10)
    cap = capfd.readouterr()
    assert "[kill] SIGKILL" in cap.err and "[kill] SIGKILL" not in cap.out
