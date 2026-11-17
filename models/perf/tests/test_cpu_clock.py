# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for tt_py_test_utils_common.perf.cpu_clock (no device needed)."""

import errno
import threading

import pytest

import tt_py_test_utils_common.perf.cpu_clock as cpu_clock
from tt_py_test_utils_common.perf.cpu_clock import UCLAMP_MAX, hold_full_cpu_clock, util_min

pytestmark = pytest.mark.skipif(util_min() is None, reason="needs Linux sched_getattr on x86_64 or aarch64")


class _Worker:
    """A thread that reports its own uclamp_min on request, so it can live across clamp and restore."""

    def __init__(self):
        self._ask, self._answered, self._stop = threading.Event(), threading.Event(), threading.Event()
        self._value = None
        self._thread = threading.Thread(target=self._run, daemon=True)  # a failed assert must not hang exit
        self._thread.start()

    def _run(self):
        while True:
            self._ask.wait()
            self._ask.clear()
            if self._stop.is_set():
                return
            self._value = util_min()
            self._answered.set()

    def util_min(self):
        self._answered.clear()
        self._ask.set()
        assert self._answered.wait(5)
        return self._value

    def stop(self):
        self._stop.set()
        self._ask.set()
        self._thread.join(5)


def _require_clamp(clock):
    if clock.clamped == 0 or clock.errors:  # no uclamp in this kernel, or syscall denied (e.g. seccomp)
        clock.restore()
        pytest.skip(clock.status)


def test_clamp_and_restore():
    before = util_min()
    workers = [_Worker()]
    try:
        existing = workers[0]
        with hold_full_cpu_clock() as clock:
            _require_clamp(clock)
            assert util_min() == UCLAMP_MAX
            assert existing.util_min() == UCLAMP_MAX
            workers.append(_Worker())  # created while clamped: inherits the clamp
            spawned = workers[1]
            assert spawned.util_min() == UCLAMP_MAX
        assert clock.restore_errors == {}
        assert util_min() == before
        assert existing.util_min() == before
        assert spawned.util_min() == before  # alive across restore, reset to the creating thread's previous value
        assert clock.restore() is True  # idempotent
    finally:
        for w in workers:
            w.stop()


def test_nested_keeps_outer_clamp():
    before = util_min()
    with hold_full_cpu_clock() as outer:
        _require_clamp(outer)
        with hold_full_cpu_clock():
            assert util_min() == UCLAMP_MAX
        assert util_min() == UCLAMP_MAX  # the inner restore puts back the outer clamp
    assert util_min() == before


# Failure paths, with the syscalls mocked (a real host can't be made to lack uclamp).


def _record_writes(monkeypatch, fail_clamp_with=None):
    writes = []

    def fake_set(libc, tid, value):
        writes.append((tid, value))
        return fail_clamp_with if value == UCLAMP_MAX and fail_clamp_with else 0

    monkeypatch.setattr(cpu_clock, "_set_util_min", fake_set)
    return writes


def test_no_uclamp_restore_is_a_no_op(monkeypatch):
    writes = _record_writes(monkeypatch, fail_clamp_with=errno.EOPNOTSUPP)
    workers = []
    try:
        clock = cpu_clock.hold_full_cpu_clock()
        assert clock.clamped == 0 and clock.errors
        workers.append(_Worker())  # created during the hold: must not get a restore write either
        writes.clear()
        assert clock.restore() is True and clock.restore_errors == {}
        assert writes == []
    finally:
        for w in workers:
            w.stop()


def test_unreadable_creator_clamps_nothing(monkeypatch):
    writes = _record_writes(monkeypatch)
    monkeypatch.setattr(cpu_clock, "_get_util_min", lambda libc, tid: None)
    clock = cpu_clock.hold_full_cpu_clock()
    assert clock.clamped == 0 and writes == []
    assert clock.restore() is True and writes == []


def test_restore_skips_unclamped_and_vanished_threads(monkeypatch):
    writes = _record_writes(monkeypatch)
    me = threading.get_native_id()
    other, gone = 10**9 + 1, 10**9 + 2  # fake tids: one fails to clamp, one is clamped then exits
    tids = [me, other, gone]
    monkeypatch.setattr(cpu_clock, "_tids", lambda: list(tids))
    real_set = cpu_clock._set_util_min
    monkeypatch.setattr(
        cpu_clock, "_set_util_min", lambda libc, tid, v: errno.EINVAL if tid == other else real_set(libc, tid, v)
    )
    monkeypatch.setattr(cpu_clock, "_get_util_min", lambda libc, tid: 7)
    clock = cpu_clock.hold_full_cpu_clock()
    assert clock.clamped == 2 and list(clock.errors.values()) == [[other]]
    tids.remove(gone)  # its tid could now belong to another process
    writes.clear()
    assert clock.restore() is True
    assert writes == [(me, 7)]  # not `other` (never clamped), not `gone` (no longer ours)
