# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The supervisor's watch: returns on events, on a stopped orchestrator at once, and on its timeout."""

import os
import threading
import time

from models.demos.common.bringup import watch as W


def _run_dir(tmp_path, pid=None, log=""):
    (tmp_path / "orchestrator.log").write_text(log)
    if pid is not None:
        (tmp_path / W.PID_FILE).write_text(f"{pid} t\n")
    return tmp_path


def test_not_running_returns_at_once_with_the_last_event(tmp_path):
    # the missed case: the orchestrator exited waiting for an approval before the watch started
    d = _run_dir(
        tmp_path, log="  [PL.0] gate PASS -> abc\n[PL.1] WAITING FOR A PERSON: approve the plan\nORCHESTRATOR_EXIT 3\n"
    )
    out = []
    t0 = time.monotonic()
    assert W.watch(d, timeout=60, poll=0.01, out=out.append) == W.NOT_RUNNING
    assert time.monotonic() - t0 < 1
    assert "WAITING FOR A PERSON" in out[0]


def test_dead_pid_counts_as_not_running(tmp_path):
    d = _run_dir(tmp_path, pid=2**22 + 12345, log="x\n")
    assert W.watch(d, timeout=60, poll=0.01, out=lambda s: None) == W.NOT_RUNNING


def test_event_after_start_only(tmp_path):
    d = _run_dir(tmp_path, pid=os.getpid(), log="  [R.1] gate PASS -> old\n")

    def later():
        time.sleep(0.05)
        with open(d / "orchestrator.log", "a") as f:
            f.write("[R.2] HF sanity (step reference)\n  [R.2] reference attempt 1: bringup-engineer\n")
            f.write("[PL.1] WAITING FOR A PERSON: approve the plan\n")

    threading.Thread(target=later).start()
    out = []
    assert W.watch(d, timeout=10, poll=0.01, out=out.append) == W.EVENT
    assert out == ["[PL.1] WAITING FOR A PERSON: approve the plan"]


def test_timeout_while_running(tmp_path):
    d = _run_dir(tmp_path, pid=os.getpid(), log="")
    out = []
    assert W.watch(d, timeout=0.05, poll=0.01, out=out.append) == W.TIMEOUT
    assert "re-arm" in out[0]


def test_event_filter():
    lines = [
        "[C.1] Component (step implement)",
        "  [C.1] gate FAIL",
        "  [C.1] implement attempt 1: bringup-engineer",
        "  [C.1] implement attempt 2: bringup-engineer",
        "  [C.1] gate PASS -> abc",
        "  [C.2] STOPPED: out of attempts",
        "  [C.3] KNOWLEDGE CONTRADICTED K-1: x",
        "  [G.s1] fix attempt 1: bringup-engineer (brief G.s1.fix.1.md)",
    ]
    assert W.events(lines) == lines[1:2] + lines[3:]
    assert W.events(lines, quiet_gates=True) == lines[3:]
