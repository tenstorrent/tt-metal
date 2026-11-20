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


# ---------------------------------------------------------------- FIX.31: a gate or a chain without the orchestrator


def test_a_live_gate_keeps_the_watch_and_its_result_wakes_it(tmp_path):
    """`gate X.3 --continue` from a shell holds the device: the watch said ORCHESTRATOR NOT RUNNING (exit 3)."""
    d = _run_dir(tmp_path, log="[X.3] old line\nORCHESTRATOR_EXIT 0\n")  # no orchestrator
    live = d / W.LIVE_DIR
    live.mkdir()
    (live / f"gate-{os.getpid()}-X.3.json").write_text(f'{{"pid": {os.getpid()}, "what": "gate X.3", "unit": "2/4"}}')

    def later():
        time.sleep(0.05)
        with open(d / W.GATES_LOG, "a") as f:
            f.write("2026-11-13 20:00:00 pid=99 [X.3] gate unit 2/4 X.3#test_ladder PASS (300 s)\n")  # routine
            f.write("2026-11-13 20:10:00 pid=99 [X.3] gate check FAIL (10 s, log x)\n")  # an agent's check
            f.write("2026-11-13 20:20:00 pid=99 [X.3] gate PASS (900 s, log x)\n")

    threading.Thread(target=later).start()
    out = []
    assert W.watch(d, timeout=10, poll=0.01, out=out.append) == W.EVENT
    assert len(out) == 1 and out[0].endswith("[X.3] gate PASS (900 s, log x)")


def test_a_live_gate_times_out_like_a_running_orchestrator(tmp_path):
    d = _run_dir(tmp_path, log="")
    (d / W.LIVE_DIR).mkdir()
    (d / W.LIVE_DIR / f"gate-{os.getpid()}-X.3.json").write_text(f'{{"pid": {os.getpid()}, "what": "gate X.3"}}')
    out = []
    assert W.watch(d, timeout=0.05, poll=0.01, out=out.append) == W.TIMEOUT
    assert "live without the orchestrator: gate X.3" in out[0]


def test_a_registered_chain_bridges_the_gap_between_its_gates(tmp_path):
    d = _run_dir(tmp_path, log="")
    p = W.register(d, os.getpid(), "chain: X.3 units")
    out = []
    assert W.watch(d, timeout=0.05, poll=0.01, out=out.append) == W.TIMEOUT  # the chain is live
    p.unlink()  # the chain ended: after the grace, not running
    out = []
    assert W.watch(d, timeout=10, poll=0.01, out=out.append, grace=0.05) == W.NOT_RUNNING
    dead = W.register(d, 2**22 + 12345, "chain: dead")
    assert dead.exists() and W.live_runs(d) == []
    assert W.main(["live", "--run-dir", str(d), "--pid", str(os.getpid()), "--what", "chain"]) == 0
    assert [r["what"] for r in W.live_runs(d)] == ["chain"]


def test_the_orchestrators_own_gate_lines_are_not_read_twice(tmp_path):
    d = _run_dir(tmp_path, pid=os.getpid(), log="")

    def later():
        time.sleep(0.05)
        with open(d / W.GATES_LOG, "a") as f:
            f.write(f"2026-11-13 20:20:00 pid={os.getpid()} [X.3] gate PASS (900 s, log x)\n")  # in its log too
        with open(d / "orchestrator.log", "a") as f:
            f.write("  [X.3] gate PASS -> abc\n")

    threading.Thread(target=later).start()
    out = []
    assert W.watch(d, timeout=10, poll=0.01, out=out.append) == W.EVENT
    assert out == ["  [X.3] gate PASS -> abc"]
