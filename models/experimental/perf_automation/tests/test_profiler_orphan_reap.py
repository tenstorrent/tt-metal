# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Issue 6: profiler daemons outlived a SUCCESSFUL run and kept holding the device.

``_execute`` starts the profiled pytest in its own session (``start_new_session=True``) precisely so
the whole process GROUP can be killed -- its docstring says the group kill exists "so orphaned
capture-release daemons die too". But the group is only killed on the stall/backstop paths. The
normal path is::

    return proc.wait(timeout=poll)

which reaps the leader and nothing else. ``tools/tracy/__main__.py`` launches ``tracy-capture`` and
``serve_wasm.py`` as children, and a daemon that outlives its parent is re-parented, not killed --
so a run that SUCCEEDS can still leave them behind. Observed in the llama3_1_8b_p150 run: 7
``tracy-capture`` and 2 ``serve_wasm`` orphans, holding the device so the next run could not open it.

A clean exit must leave nothing behind. The leader's own exit code is unaffected -- reaping happens
after it, and never changes what _execute returns.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _probes():
    from agent import probes

    return probes


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# A leader that spawns a long-lived daemon child, then exits 0 immediately -- the shape of a
# successful tracy run that leaves tracy-capture behind.
_LEADER = (
    "import subprocess,sys,time\n"
    "c = subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'])\n"
    "print(c.pid, flush=True)\n"
    "sys.exit(0)\n"
)


def _run_leader(tmp_path, extra=""):
    m = _probes()
    log = tmp_path / "run.log"
    script = tmp_path / "leader.py"
    script.write_text(_LEADER + extra)
    rc = m._execute([sys.executable, str(script)], tmp_path, dict(os.environ), 120, log, stall_timeout_s=60)
    child_pid = int(log.read_text().split()[0])
    return rc, child_pid


def test_reap_helper_exists():
    m = _probes()
    if getattr(m, "_reap_process_group", None) is None:
        pytest.fail(
            "probes has no _reap_process_group: _execute only kills the process group on the "
            "stall/backstop paths, so a SUCCESSFUL run leaves tracy-capture / serve_wasm daemons "
            "alive holding the device (7 + 2 orphans observed on llama3_1_8b_p150)."
        )


def test_successful_run_leaves_no_orphan(tmp_path):
    test_reap_helper_exists()
    rc, child_pid = _run_leader(tmp_path)
    assert rc == 0, "the leader's exit code must be unaffected by reaping"
    time.sleep(0.4)
    assert not _alive(child_pid), (
        f"daemon child {child_pid} survived a successful run -- it holds the device and the next " "run cannot open it"
    )


def test_exit_code_is_preserved_on_failure(tmp_path):
    test_reap_helper_exists()
    m = _probes()
    log = tmp_path / "f.log"
    script = tmp_path / "fail.py"
    script.write_text(
        "import subprocess,sys\n"
        "c = subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'])\n"
        "print(c.pid, flush=True)\n"
        "sys.exit(7)\n"
    )
    rc = m._execute([sys.executable, str(script)], tmp_path, dict(os.environ), 120, log, stall_timeout_s=60)
    assert rc == 7, "reaping must not alter a non-zero exit code"
    child_pid = int(log.read_text().split()[0])
    time.sleep(0.4)
    assert not _alive(child_pid), "orphan survived a FAILED run"


def test_run_with_no_children_is_unaffected(tmp_path):
    test_reap_helper_exists()
    m = _probes()
    log = tmp_path / "q.log"
    script = tmp_path / "quiet.py"
    script.write_text("print('done', flush=True)\n")
    rc = m._execute([sys.executable, str(script)], tmp_path, dict(os.environ), 120, log, stall_timeout_s=60)
    assert rc == 0
    assert "done" in log.read_text()


def test_reap_never_touches_our_own_process_group():
    """The reaper must only ever kill the CHILD session it created. Killing our own group would
    take out the optimize run itself."""
    m = _probes()
    test_reap_helper_exists()
    own = os.getpgid(0)
    m._reap_process_group(own)  # must be a no-op, not suicide
    assert _alive(os.getpid())


def test_reap_of_a_dead_group_is_a_noop():
    m = _probes()
    test_reap_helper_exists()
    p = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    p.wait()
    m._reap_process_group(p.pid)  # group already gone


@pytest.mark.parametrize("bad", [0, -1, None, "x", 999999999])
def test_reap_hostile_pgid_never_raises(bad):
    m = _probes()
    test_reap_helper_exists()
    m._reap_process_group(bad)
