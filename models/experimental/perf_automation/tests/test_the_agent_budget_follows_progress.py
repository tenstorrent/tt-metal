# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The agent's overall budget is a clock for a quiet agent, not for one whose tree is still moving.

It used to end the agent the moment the clock ran out, even while the run_perf_test attempts it had
started were still writing output. Now the budget extends while probes.ProgressWatch reports
progress, up to the shared hard ceiling, and ends as before once the tree goes quiet.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA.parent.parent.parent))

from models.experimental.perf_automation.agent import perf_test_agent as pta  # noqa: E402


class _Watch:
    def __init__(self, moving):
        self.moving = moving

    def moved(self, now, last):
        return self.moving

    def note_progress(self, now, last):
        pass

    def limit(self):
        return 0.3


def _budget(budget_s, watch, mult=3):
    b = pta._AgentBudget(os.getpid(), None, budget_s)
    b._watch, b._mult = watch, mult
    return b


def test_a_quiet_agent_ends_at_its_budget():
    b = _budget(0.2, _Watch(False))
    assert not b.expired()
    time.sleep(0.5)
    assert b.expired()


def test_a_moving_agent_outlives_its_budget_but_not_the_hard_ceiling():
    b = _budget(0.2, _Watch(True), mult=3)
    time.sleep(0.3)
    assert not b.expired(), "killed a tree that was still making progress"
    time.sleep(0.4)
    assert b.expired(), "the hard ceiling no longer ends it"


def test_without_a_progress_signal_it_is_the_plain_budget():
    b = _budget(0.2, None)
    time.sleep(0.3)
    assert b.expired()


def test_real_tree_writing_its_log_is_kept_and_a_silent_one_is_not(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_VALIDATE_STALL_SEC", "2")
    log = tmp_path / "agent.log"
    writer = "import time\nf=open(%r,'w',buffering=1)\nwhile True:\n f.write('x\\n'); time.sleep(0.2)\n" % str(log)
    proc = subprocess.Popen([sys.executable, "-c", writer], start_new_session=True)
    try:
        b = pta._AgentBudget(proc.pid, log, 1)
        b._mult = 20
        end = time.monotonic() + 4
        while time.monotonic() < end:
            assert not b.expired(), "a tree still writing its log was ended"
            time.sleep(0.3)
    finally:
        proc.kill()
        proc.wait()
    quiet = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        b = pta._AgentBudget(quiet.pid, tmp_path / "none.log", 1)
        b._mult = 20
        end = time.monotonic() + 10
        while time.monotonic() < end and not b.expired():
            time.sleep(0.3)
        assert b.expired(), "a silent tree outlived its budget"
    finally:
        quiet.kill()
        quiet.wait()
