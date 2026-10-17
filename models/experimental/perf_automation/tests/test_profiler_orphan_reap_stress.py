# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""STRESS for issue 6: a finished profiler run must leave zero survivors.

The invariant: after _execute returns, no process from the run's session is alive, and the exit
code is whatever the leader returned.

  s1  the observed llama shape -- 7 "tracy-capture" + 2 "serve_wasm" daemons, all reaped
  s2  deep grandchild chains and double-forked daemons (the classic re-parenting escape)
  s3  30 back-to-back runs leak nothing cumulatively (the soak the plan asks for)
  s4  exit codes survive reaping across the full range, including signals
  s5  the reaper is scoped: sibling sessions and our own group are never touched
  s6  daemons that ignore SIGTERM still die (SIGKILL), and already-dead groups are a no-op
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


def _wait_gone(pids, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not any(_alive(p) for p in pids):
            return True
        time.sleep(0.05)
    return False


def _run(tmp_path, body, name="leader.py", rc_expect=None, timeout_s=120):
    m = _probes()
    log = tmp_path / (name + ".log")
    script = tmp_path / name
    script.write_text(body)
    rc = m._execute([sys.executable, str(script)], tmp_path, dict(os.environ), timeout_s, log, stall_timeout_s=60)
    pids = [int(x) for x in log.read_text().split() if x.isdigit()]
    if rc_expect is not None:
        assert rc == rc_expect, f"exit code {rc} != {rc_expect}"
    return rc, pids


# --------------------------------------------------------------------------- s1
def test_s1_the_observed_llama_orphan_shape(tmp_path):
    """7 tracy-capture + 2 serve_wasm, exactly what the run left behind."""
    body = (
        "import subprocess,sys\n"
        "pids=[]\n"
        "for i in range(9):\n"
        "    c = subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'])\n"
        "    pids.append(c.pid)\n"
        "print(' '.join(str(p) for p in pids), flush=True)\n"
        "sys.exit(0)\n"
    )
    rc, pids = _run(tmp_path, body, rc_expect=0)
    assert len(pids) == 9
    assert _wait_gone(pids), f"survivors: {[p for p in pids if _alive(p)]}"


# --------------------------------------------------------------------------- s2
def test_s2_grandchildren_are_reaped(tmp_path):
    body = (
        "import subprocess,sys\n"
        "c = subprocess.Popen([sys.executable,'-c',\n"
        '  "import subprocess,sys,time;"\n'
        "  \"g=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)']);\"\n"
        '  "print(g.pid, flush=True); time.sleep(300)"], stdout=subprocess.PIPE, text=True)\n'
        "gp = c.stdout.readline().strip()\n"
        "print(c.pid, gp, flush=True)\n"
        "sys.exit(0)\n"
    )
    rc, pids = _run(tmp_path, body, rc_expect=0)
    assert len(pids) == 2, f"expected child+grandchild, got {pids}"
    assert _wait_gone(pids), f"survivors: {[p for p in pids if _alive(p)]}"


def test_s2_double_forked_daemon_is_reaped(tmp_path):
    """A double-fork re-parents to init -- the classic way a daemon escapes its parent. It stays in
    the SESSION though, which is why _execute uses start_new_session and why group-kill works."""
    body = (
        "import os,sys,time\n"
        "r,w = os.pipe()\n"
        "if os.fork() == 0:\n"
        "    if os.fork() == 0:\n"
        "        os.write(w, str(os.getpid()).encode())\n"
        "        time.sleep(300)\n"
        "    os._exit(0)\n"
        "os.close(w)\n"
        "pid = os.read(r, 32).decode()\n"
        "print(pid, flush=True)\n"
        "sys.exit(0)\n"
    )
    rc, pids = _run(tmp_path, body, rc_expect=0)
    assert pids and _wait_gone(pids), f"double-forked daemon survived: {pids}"


# --------------------------------------------------------------------------- s3
def test_s3_thirty_runs_leak_nothing_cumulatively(tmp_path):
    body = (
        "import subprocess,sys\n"
        "c = subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'])\n"
        "print(c.pid, flush=True)\n"
        "sys.exit(0)\n"
    )
    all_pids = []
    for i in range(30):
        _rc, pids = _run(tmp_path, body, name=f"soak{i}.py", rc_expect=0)
        all_pids += pids
    assert _wait_gone(
        all_pids, timeout=10
    ), f"{len([p for p in all_pids if _alive(p)])} of 30 runs' daemons accumulated"


# --------------------------------------------------------------------------- s4
@pytest.mark.parametrize("code", [0, 1, 2, 7, 42, 127])
def test_s4_exit_codes_survive_reaping(tmp_path, code):
    body = (
        "import subprocess,sys\n"
        "c = subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'])\n"
        "print(c.pid, flush=True)\n"
        f"sys.exit({code})\n"
    )
    rc, pids = _run(tmp_path, body, name=f"rc{code}.py")
    assert rc == code, f"reaping changed the exit code: {rc} != {code}"
    assert _wait_gone(pids)


def test_s4_leader_killed_by_signal_still_reaps(tmp_path):
    body = (
        "import subprocess,sys,os,signal\n"
        "c = subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'])\n"
        "print(c.pid, flush=True)\n"
        "os.kill(os.getpid(), signal.SIGTERM)\n"
    )
    rc, pids = _run(tmp_path, body)
    assert rc != 0
    assert _wait_gone(pids), "a signal-killed leader left its daemon behind"


# --------------------------------------------------------------------------- s5
def test_s5_sibling_session_is_untouched(tmp_path):
    """Reaping one run must not disturb an unrelated concurrent run."""
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        body = (
            "import subprocess,sys\n"
            "c = subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'])\n"
            "print(c.pid, flush=True)\n"
            "sys.exit(0)\n"
        )
        _rc, pids = _run(tmp_path, body, rc_expect=0)
        assert _wait_gone(pids)
        assert _alive(bystander.pid), "reaping killed an unrelated session"
    finally:
        bystander.kill()
        bystander.wait()


def test_s5_own_group_is_refused():
    m = _probes()
    assert m._reap_process_group(os.getpgid(0)) == []
    assert _alive(os.getpid())


def test_s5_returns_the_pids_it_killed(tmp_path):
    m = _probes()
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    killed = m._reap_process_group(p.pid)
    assert p.pid in killed
    p.wait(timeout=5)


# --------------------------------------------------------------------------- s6
def test_s6_sigterm_ignoring_daemon_still_dies(tmp_path):
    body = (
        "import subprocess,sys\n"
        "c = subprocess.Popen([sys.executable,'-c',\n"
        "  'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)'])\n"
        "print(c.pid, flush=True)\n"
        "sys.exit(0)\n"
    )
    _rc, pids = _run(tmp_path, body, rc_expect=0)
    assert _wait_gone(pids), "a SIGTERM-ignoring daemon survived; the reaper must use SIGKILL"


def test_s6_dead_group_is_a_noop():
    m = _probes()
    p = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    p.wait()
    assert m._reap_process_group(p.pid) == []


@pytest.mark.parametrize("bad", [0, -1, -999, None, "", "x", [], {}, 10**9])
def test_s6_hostile_pgid_never_raises(bad):
    m = _probes()
    assert m._reap_process_group(bad) == []


def test_s6_members_of_a_nonexistent_group_is_empty():
    m = _probes()
    assert m._pgroup_members(10**9) == []
