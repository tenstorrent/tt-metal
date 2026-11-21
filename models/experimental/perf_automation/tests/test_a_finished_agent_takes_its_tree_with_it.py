# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""An agent that finishes normally must not leave workers behind, and a supervisor must not race them.

WHAT HAPPENED, 2026-08-16. Attempt 1 of an optimize run gave up on perf-test generation and exited
rc=1. Seventy-seven minutes later its process tree was still running:

    611424  1:17:07  orchestrator          own session
     621626  1:09:54  device subprocess    own session
      636105    37:53  perf-test agent     own session, 0 device fds

The supervisor had already started attempt 2 into the same board. Two runs driving one board took the
ARC cores down: `tt-smi -r` failed with "ARC core (8, 0) failed to start", twice, and both chips read
65535999 (the 0xFFFF no-reading sentinel) instead of a temperature -- which then made every thermal
gate wait its full 900s and measure hot. Killing the tree by hand and running the IDENTICAL reset
brought all four chips back at 67-70C. So it was contention, not hardware.

THREE MECHANISMS EXISTED AND NONE APPLIED.

    cli.py:_kill_process_tree        /proc walk + per-pid kill   -> agent calls, pytest
    cc_harness.py:_kill_agent_tree   same                        -> cc_harness only
    probes.py:_kill_tree             pids AND every pgid found   -> hang_probe, _execute

perf_test_agent.py, which spawned the leaked agent, used none of them: one `os.killpg` on the agent's
own group, and only under `except TimeoutExpired`. The claude CLI puts its workers in their own
sessions, so the group kill could never have reached them -- and the agent did not time out, it
finished its turn budget, so the branch never ran at all.

The supervisor used none of them either. It called `_sp.run(...)` and treated the return as the
attempt ending, then reclaimed DEVICE HOLDERS -- and a worker between device operations holds nothing.

WHY THE /proc WALK IS NOT ENOUGH ON ITS OWN. `_descendant_pids` reads PPIDs, so it can only see a
tree whose root is alive. A caller that WAITS for a process rather than killing it has nothing left
to walk the moment it returns: the children are reparented to init and the link is gone. Both fixes
therefore snapshot the tree WHILE it runs and reap from that snapshot afterwards.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:  # noqa: BLE001
        return False


def _wait_gone(pid, timeout=5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def _spawn_detaching_parent(tmp_path):
    """A parent that spawns a worker in ITS OWN SESSION and then exits -- the claude CLI's shape.

    The worker is reparented to init, keeps running, and belongs to no group the caller can signal.
    Its pid is written to a file because after the parent exits nothing can name it.
    """
    pidfile = tmp_path / "worker.pid"
    src = (
        "import os,subprocess,sys,time\n"
        "p = subprocess.Popen([sys.executable,'-c','import time; time.sleep(600)'], start_new_session=True)\n"
        "open(%r,'w').write(str(p.pid))\n" % str(pidfile)
    )
    parent = subprocess.Popen([sys.executable, "-c", src], start_new_session=True)
    for _ in range(200):
        if pidfile.is_file() and pidfile.read_text().strip():
            break
        time.sleep(0.05)
    worker = int(pidfile.read_text().strip())
    return parent, worker


# ------------------------------------------------------------------ the primitive


def test_the_reaper_kills_a_pid_it_can_no_longer_walk_to(tmp_path):
    """THE HALF THE /proc WALK CANNOT DO. Once the root exits, the descendant is reparented and
    unreachable by PPID -- but a caller that remembered it can still say so."""
    from agent.probes import _descendant_pids, _kill_tree

    parent, worker = _spawn_detaching_parent(tmp_path)
    remembered = set(_descendant_pids(parent.pid)) | {worker}
    parent.wait(timeout=10)

    assert _alive(worker), "the worker did not outlive its parent; the test proves nothing"
    assert worker not in _descendant_pids(parent.pid), "the walk still reaches a reparented orphan"

    _kill_tree(parent.pid, extra=remembered)
    assert _wait_gone(worker), "a remembered orphan survived the reap"


def test_the_reaper_never_kills_its_own_group():
    """`extra` is caller-supplied, so the walk's structural safety no longer holds. A snapshot pid
    whose group happens to be ours would take out the run doing the reaping."""
    from agent.probes import _kill_tree

    _kill_tree(os.getpid(), extra=[os.getpid()])
    assert _alive(os.getpid())


def test_the_reaper_survives_pids_that_are_already_gone(tmp_path):
    """Reaping runs on the recovery path; a stale snapshot entry must degrade, never raise."""
    from agent.probes import _kill_tree

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=10)
    _kill_tree(dead.pid, extra=[dead.pid, 999999, -1, 0])


# ------------------------------------------------------------------ the perf-test agent


def _agent_src() -> str:
    return (_PA / "agent" / "perf_test_agent.py").read_text()


def test_the_agent_reaps_on_every_exit_not_only_on_timeout():
    """THE BUG: the kill lived under `except TimeoutExpired`, so an agent that finished its turn
    budget -- the NORMAL ending -- left everything it had spawned running."""
    src = _agent_src()
    i = src.index("proc = subprocess.Popen(cmd")
    body = src[i : i + 3000]
    assert "finally:" in body, "the reap is still conditional on how the agent ended"
    fin = body[body.index("finally:") :]
    assert "_reap(" in fin.split("\n\n")[0] or "_reap(" in fin[:600], "nothing is reaped on the normal exit path"


def test_the_agent_snapshots_the_tree_while_it_is_alive():
    """Reaping after the fact is useless without this: the pids are unreachable by then."""
    src = _agent_src()
    i = src.index("proc = subprocess.Popen(cmd")
    body = src[i : i + 3000]
    assert "_desc(proc.pid)" in body, "the tree is never snapshotted"
    assert body.index("_desc(proc.pid)") < body.index("_reap("), "the snapshot is taken after the reap"


def test_the_agent_no_longer_relies_on_a_bare_group_kill():
    """killpg cannot reach a worker that made its own session, which is what the claude CLI does."""
    src = _agent_src()
    i = src.index("proc = subprocess.Popen(cmd")
    body = src[i : i + 3000]
    assert "os.killpg" not in body, "the group kill is back as the agent's only reap"


# ------------------------------------------------------------------ the supervisor


def _supervisor_src() -> str:
    p = _PA.parent.parent.parent / "scripts" / "tt_hw_planner" / "commands" / "optimize.py"
    return p.read_text() if p.is_file() else ""


def test_the_supervisor_tracks_the_tree_rather_than_only_its_child():
    """`_sp.run` returning proves ONE process ended. The attempt is its whole tree."""
    src = _supervisor_src()
    if not src:
        return
    i = src.index("def _run_attempt(")
    body = src[i : src.index("\n        for _n in range(", i)]
    assert "_desc(_p.pid)" in body, "the supervisor still watches only its direct child"
    assert "_reap(" in body, "the supervisor never kills what the attempt left behind"


def test_the_supervisor_refuses_to_restart_over_a_survivor():
    """Starting attempt N+1 while attempt N still runs is what took the ARC cores down. A process
    that survives SIGKILL is in D-state on the device -- the one case where another attempt is the
    worst possible move."""
    src = _supervisor_src()
    if not src:
        return
    i = src.index("def _run_attempt(")
    body = src[i : src.index("\n        for _n in range(", i)]
    assert "survived SIGKILL" in body, "a survivor no longer stops the restart"
    j = src.index("for _n in range(_max + 1):")
    assert "if _stuck:" in src[j : j + 400], "the loop does not act on the refusal"


def test_the_supervisor_reaps_before_it_resets_the_device():
    """Ordering, and it is not cosmetic: the 2026-08-16 reset FAILED with 'ARC core (8, 0) failed to
    start' while the leaked tree still held the board, and the identical command succeeded once the
    tree was dead."""
    src = _supervisor_src()
    if not src:
        return
    assert src.index("def _run_attempt(") < src.index("_reclaim_device as _rcl")


def test_the_supervisor_stops_calling_every_failure_a_native_crash():
    """It printed 'likely native crash / device wedge' for a perf-test generation failure that never
    touched the device, and that fixed string sent three investigations to the wrong subsystem."""
    src = _supervisor_src()
    if not src:
        return
    i = src.index("orchestrator exited rc=")
    assert "likely native crash" not in src[i : i + 300]
