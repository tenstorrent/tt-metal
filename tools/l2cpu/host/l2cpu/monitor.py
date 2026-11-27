# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""Heartbeat / error watch thread for the L2CPU firmware (generic part).

Every `interval` (100 ms) it reads, through an L2cpuCtl: the four heartbeats and the first-error word, plus any
application checks (`extra_checks`: callables returning a failure reason or None). It fails when
  - the firmware error word is non-zero (trap, worker dead, work timeout, application error), or
  - a hart that is not parked has a heartbeat that has not changed for `stall_s` (2 s), or
  - an application check reports a failure.
On failure it calls `on_fail(reason)`. With `exit_code` set (default 17) it then prints the hart states and the
firmware log ring and terminates the process with os._exit: the main thread may be blocked in a device wait that
spins until its own timeout, and closing the device from there could hang. os._exit does not touch the device; the
next run must start with a chip reset (scripts/l2cpu_run.sh does). With exit_code=None it only records the
failure (`.failed`) and returns, so the caller can attempt a restart (ctl.restart) itself.

All host accesses (main thread and monitor) are serialized by a lock installed on the backend.
"""
from __future__ import annotations

import os
import sys
import threading
import time

from . import layout as A


def install_lock(hw):
    """Serialize every NoC access of this L2cpuHw (its backend) behind one lock. Idempotent."""
    b = hw.b
    if getattr(b, "_l2cpu_lock", None) is not None:
        return b._l2cpu_lock
    lock = threading.RLock()
    for name in ("read", "read32", "write", "write32"):
        f = getattr(b, name)

        def locked(*a, _f=f, **k):
            with lock:
                return _f(*a, **k)

        setattr(b, name, locked)
    b._l2cpu_lock = lock
    return lock


class L2cpuMonitor:
    def __init__(self, ctl, interval=0.1, stall_s=2.0, exit_code=17, log=None, on_fail=None, extra_checks=()):
        self.ctl = ctl
        self.interval, self.stall_s, self.exit_code = interval, stall_s, exit_code
        self.log = log or (
            lambda *a: print("[l2cpu-monitor %s]" % time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)
        )
        self.on_fail = on_fail
        self.extra_checks = list(extra_checks)
        install_lock(ctl.hw)
        self._stop = threading.Event()
        self.thread = None
        self.samples = 0
        self.max_hb_gap = [0.0] * 4
        self.failed = None

    def start(self):
        self._last_hb = self.ctl.heartbeats()
        self._last_change = [time.time()] * 4
        self.thread = threading.Thread(target=self._run, name="l2cpu-monitor", daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=2)
        return dict(
            samples=self.samples, max_heartbeat_gap_s=[round(x, 3) for x in self.max_hb_gap], failed=self.failed
        )

    def check(self):
        """One sample. Returns a failure reason or None."""
        ctl = self.ctl
        now = time.time()
        err = ctl.error()
        if err:
            return f"firmware error {err}"
        hb = ctl.heartbeats()
        parked = [r["state"] == A.L2CPU_STATE_PARKED for r in ctl.records()] if ctl.ready() else [False] * 4
        for h in range(4):
            if hb[h] != self._last_hb[h] or parked[h]:
                self.max_hb_gap[h] = max(self.max_hb_gap[h], now - self._last_change[h])
                self._last_hb[h], self._last_change[h] = hb[h], now
            elif now - self._last_change[h] > self.stall_s:
                return f"hart {h} heartbeat stuck at {hb[h]} for {now - self._last_change[h]:.1f} s"
        for chk in self.extra_checks:
            why = chk()
            if why:
                return why
        self.samples += 1
        return None

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                why = self.check()
            except Exception as e:  # noqa: BLE001  (device unreachable is a failure too)
                why = f"monitor read failed: {e!r}"
            if why:
                self.fail(why)
                return

    def fail(self, why):
        self.failed = why
        self.log("FAIL:", why)
        if self.on_fail:
            try:
                self.on_fail(why)
            except Exception:  # noqa: BLE001
                pass
        if self.exit_code is None:
            return
        try:
            for h in range(4):
                self.log(f"hart {h}: {self.ctl.hart_state(h)} record {self.ctl.record(h)}")
            self.log("---- firmware log ring ----\n" + self.ctl.log_text(0)[0][-8000:])
        except Exception as e:  # noqa: BLE001
            self.log("dump failed:", repr(e))
        self.log(f"exiting with code {self.exit_code}; the next run must reset the chip")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(self.exit_code)
