"""The stall check counts the work of a child that moved itself into its own session.

tracy launches the profiled workload with preexec_fn=os.setsid, so the workload leaves the process
group the supervisor created and watches. The progress count summed that group only: the launcher and
the capture tool, idle during a device-profiler read-back, while the test doing the reading moved
13,240 syscalls and 90 MB in 30 s (WH Galaxy, 2026-09-27) -- and the stall check killed it as "no
forward progress", three times running. The kill already walks the leader's whole tree; these pin
that the progress count now sees the same tree, and nothing outside it.

Real processes, no mocks: the question is what /proc attributes to whom.
"""

import os
import subprocess
import sys
import time

import pytest

from agent import probes

_BUSY = "import sys, time\nt = time.time()\nwhile time.time() - t < 30:\n    open('/proc/self/stat').read()\n"
# leader: start the busy child in a NEW session (exactly what tracy does), then sit idle waiting on it
_LEADER = (
    "import os, subprocess, sys, time\n"
    "c = subprocess.Popen([sys.executable, '-c', %r], preexec_fn=os.setsid)\n"
    "print(c.pid, flush=True)\n"
    "c.wait()\n" % _BUSY
)

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs /proc")


@pytest.fixture
def run():
    leader = subprocess.Popen(
        [sys.executable, "-c", _LEADER], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    child = int(leader.stdout.readline())
    yield leader, child
    for pid in (child, leader.pid):
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    leader.wait(timeout=10)


def _sample(pgid, wait=1.0):
    a = probes._pgroup_io_counters(pgid)
    time.sleep(wait)
    return a, probes._pgroup_io_counters(pgid)


def test_the_child_really_left_the_group(run):
    leader, child = run
    assert os.getpgid(child) != leader.pid, "the fixture must reproduce tracy's setsid split"
    assert os.getsid(child) == child


def test_its_work_is_counted_as_the_runs_progress(run):
    leader, child = run
    a, b = _sample(leader.pid)
    assert b[0] > a[0], "a busy child in its own session is the run making progress"


def test_the_watch_says_moved_while_only_the_child_works(run):
    leader, child = run
    watch = probes.ProgressWatch(leader.pid, None, 600)
    time.sleep(1.0)
    assert watch.moved(time.monotonic(), time.monotonic(), leader.pid) is True


def test_a_process_outside_the_tree_is_not_counted(run):
    leader, child = run
    stranger = subprocess.Popen([sys.executable, "-c", _BUSY], start_new_session=True)
    try:
        time.sleep(0.5)
        members_before = probes._pgroup_io_counters(stranger.pid)
        assert members_before[0] > 0, "sanity: the stranger's own group is measurable"
        # the stranger is neither in the leader's group nor its descendant
        assert stranger.pid not in probes._descendant_pids(leader.pid)
        assert os.getpgid(stranger.pid) != leader.pid
    finally:
        stranger.kill()
        stranger.wait(timeout=10)


def test_a_leader_that_has_exited_counts_its_group_alone():
    # an exited leader has no descendants left (they are reparented), so the sum is the group's
    gone = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    gone.wait(timeout=10)
    assert probes._pgroup_io_counters(gone.pid) == (0, 0)


def test_an_unusable_group_id_still_answers():
    assert probes._pgroup_io_counters("not-a-pid") == (0, 0)
