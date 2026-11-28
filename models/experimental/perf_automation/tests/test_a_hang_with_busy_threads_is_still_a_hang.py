# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Counters that move under a frozen stack are not progress.

MiniCPM5-2B, 2026-10-12. A profile's test sat in one `to_torch` for three hours on a hung chip. Its
Python stack was byte-identical across samples, but its background threads moved ~200 syscalls every
20 s, and ProgressWatch only looked at the stack once the counters went quiet -- which they never did.
Every poll read as progress, so the stall check never started and only the 12 h ceiling stood behind
it.

What must still hold: a stack that moves, a log that grows, a process whose stack cannot be read (a
compiler, the capture tool, a box without py-spy), or a new process all keep the run alive.
"""

import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

import agent.probes as P  # noqa: E402

_STEP = P._STACK_EVERY_S + 1.0


class _Tree:
    """A fake run: per-pid io counters that tick on every read, and a stack per pid."""

    def __init__(self, stacks):
        self.stacks = dict(stacks)
        self.io = {pid: 0 for pid in stacks}
        self.ticking = set(stacks)

    def counters(self, pgid):
        for pid in self.ticking:
            self.io[pid] += 200
        return {pid: (n, 0) for pid, n in self.io.items()}

    def summed(self, pgid):
        snap = self.counters(pgid)
        return sum(c for c, _ in snap.values()), 0


@pytest.fixture
def tree(monkeypatch):
    t = _Tree({101: "to_torch frozen"})
    monkeypatch.setattr(P, "_tree_io_counters", t.counters)
    monkeypatch.setattr(P, "_pgroup_io_counters", t.summed)
    monkeypatch.setattr(P, "_stack_fingerprint", lambda pid: t.stacks.get(pid, ""))
    return t


def _polls(watch, n, t0=1000.0, pid=101):
    return [watch.moved(t0 + i * _STEP, t0, pid) for i in range(1, n + 1)]


def test_busy_counters_under_a_frozen_stack_stop_counting(tree):
    watch = P.ProgressWatch(1, None, 600)
    seen = _polls(watch, 4)
    # the first sample has nothing to compare against, so it gives the counters their word
    assert seen[0] is True
    assert seen[1:] == [False, False, False], "a hang with chatty threads still looks like progress"


def test_counters_between_samples_do_not_reset_the_clock_once_frozen(tree):
    watch = P.ProgressWatch(1, None, 600)
    _polls(watch, 2)
    t = 1000.0 + 2 * _STEP
    # polls closer together than the stack interval: counters tick, no new stack sample is taken
    assert [watch.moved(t + k, 1000.0, 101) for k in (1, 2, 3)] == [False, False, False]


def test_a_moving_stack_is_progress(tree):
    watch = P.ProgressWatch(1, None, 600)
    seen = []
    for i in range(1, 5):
        tree.stacks[101] = "decode step %d" % i
        seen.append(watch.moved(1000.0 + i * _STEP, 1000.0, 101))
    assert seen == [True, True, True, True]


def test_a_frozen_run_that_moves_again_is_alive_again(tree):
    watch = P.ProgressWatch(1, None, 600)
    assert _polls(watch, 2)[-1] is False
    tree.stacks[101] = "next layer"
    assert watch.moved(1000.0 + 3 * _STEP, 1000.0, 101) is True


def test_an_unreadable_stack_keeps_the_counters_word(tree):
    """No py-spy, a compiler, the capture tool: exactly the old behaviour."""
    tree.stacks[101] = ""
    watch = P.ProgressWatch(1, None, 600)
    assert _polls(watch, 4) == [True, True, True, True]


def test_one_unreadable_mover_is_enough_to_stay_alive(tree):
    tree.stacks[202] = ""  # e.g. a g++ child of a kernel compile
    tree.io[202] = 0
    tree.ticking.add(202)
    watch = P.ProgressWatch(1, None, 600)
    assert _polls(watch, 4) == [True, True, True, True]


def test_a_new_process_is_not_frozen(tree):
    watch = P.ProgressWatch(1, None, 600)
    assert _polls(watch, 2)[-1] is False
    tree.stacks[303] = "fresh worker"
    tree.io[303] = 0
    tree.ticking.add(303)
    assert watch.moved(1000.0 + 3 * _STEP, 1000.0, 101) is True


def test_an_idle_member_with_a_still_stack_does_not_matter(tree):
    """The tracy launcher waits on its child with a still stack and still counters. Only processes
    that MOVE the counters are asked about their stack."""
    tree.stacks[404] = "launcher waiting"
    tree.io[404] = 0  # never ticks
    watch = P.ProgressWatch(1, None, 600)
    seen = []
    for i in range(1, 5):
        tree.stacks[101] = "step %d" % i
        seen.append(watch.moved(1000.0 + i * _STEP, 1000.0, 101))
    assert seen == [True, True, True, True]


def test_log_growth_always_counts(tree, tmp_path):
    log = tmp_path / "run.log"
    log.write_text("x")
    watch = P.ProgressWatch(1, log, 600)
    assert _polls(watch, 2)[-1] is False
    log.write_text("xy")
    assert watch.moved(1000.0 + 3 * _STEP, 1000.0, 101) is True


def test_without_a_pid_the_counters_decide_as_before(tree):
    watch = P.ProgressWatch(1, None, 600)
    assert [watch.moved(1000.0 + i * _STEP, 1000.0, None) for i in range(1, 4)] == [True, True, True]


def test_the_stall_clock_fires_on_the_real_loop_shape(tree):
    """End to end over the loop the supervisors run: last_progress only advances on moved()."""
    watch = P.ProgressWatch(1, None, 120)
    start = last = 1000.0
    fired_at = None
    for i in range(1, 200):
        now = start + i * 5.0
        if watch.moved(now, last, 101):
            watch.note_progress(now, last)
            last = now
        if now - last >= watch.limit():
            fired_at = now - start
            break
    assert fired_at is not None, "the stall check never fires on a frozen stack with chatty threads"
    assert fired_at <= 120 + 2 * _STEP


def test_the_summed_counters_are_the_per_process_ones_added_up(monkeypatch):
    monkeypatch.setattr(P, "_tree_io_counters", lambda pgid: {1: (3, 10), 2: (4, 20)})
    assert P._pgroup_io_counters(1) == (7, 30)


def test_the_stack_read_carries_locals_does_not_pause_and_keeps_outer_frames(monkeypatch):
    calls = []

    class _R:
        returncode = 0

        def __init__(self, out):
            self.stdout = out

    inner = "    to_torch (ttnn/operations/core.py:460)\n" + "        x: 1\n" * 400
    outs = iter([inner + "    main (loop.py:9)\n        step: 1\n", inner + "    main (loop.py:9)\n        step: 2\n"])

    def _run(cmd, **kw):
        calls.append(cmd)
        return _R(next(outs))

    monkeypatch.setattr(P, "_py_spy_exe", lambda: "/x/py-spy")
    import subprocess

    monkeypatch.setattr(subprocess, "run", _run)
    a, b = P._stack_fingerprint(7), P._stack_fingerprint(7)
    assert "--locals" in calls[0] and "--nonblocking" in calls[0]
    assert a and b and a != b, "a change in an outer frame past the old 2000-char cut is lost"


def test_py_spy_beside_the_interpreter_is_found(monkeypatch, tmp_path):
    import shutil

    exe = tmp_path / "py-spy"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    assert P._py_spy_exe() == str(exe)


def test_no_py_spy_anywhere_is_empty_not_an_error(monkeypatch, tmp_path):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    assert P._py_spy_exe() == ""
    assert P._stack_fingerprint(1) == ""
