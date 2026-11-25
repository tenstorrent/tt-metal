# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Where a serving process's wall time goes, by stage, printed every few seconds while there is work.

Every stage is a name; its first dotted part says what kind of time it is:

* ``compute.*`` -- the device computing (or a host thread blocked on the device's output, which includes the
  device-to-host read: from the host the two cannot be told apart);
* ``idle.*`` -- nothing to do;
* ``wait.*`` -- a request waiting on something else (a slot, a seat, the scheduler, the prefill server);
* anything else -- overhead: HTTP, JSON, tokenizing and detokenizing, host dispatch, wire transfer, KV loads, Python.

``with OVERHEAD.time("http.sse_write"): ...`` or ``OVERHEAD.add("prefill.kv_load", seconds)`` records one stage. A
thread that calls :meth:`Overhead.loop_iteration` around each pass of its loop also gets ``<loop>.untracked``: the
part of the pass no stage of its own covers (the loop's own Python). Timers nest; only the outermost one of a thread
counts against its loop's untracked time.

Every ``DEEPSEEK_OVERHEAD_PERIOD`` seconds (10) with something recorded a table goes to the log: per stage the count,
the total, its share of the window's wall time, the mean and the max, and how many threads recorded it (several
scheduler threads, one per instance, each own the whole window: a share of 500% over five of them is 100% each).
A stage named under another (``sched.admit.render`` under ``sched.admit``) is a part of it: listed, not totalled
twice.
``DEEPSEEK_OVERHEAD=0`` turns it off.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time

from loguru import logger

_KINDS = ("compute", "idle", "wait")


def kind(name: str) -> str:
    head = name.split(".", 1)[0]
    return head if head in _KINDS else "overhead"


class Overhead:
    def __init__(self, enabled: bool = True, period: float = 10.0):
        self.enabled = enabled
        self.period = period
        self._lock = threading.Lock()
        self._stats: dict[str, list] = {}  # name -> [count, total, max, thread ids]
        self._t0 = time.perf_counter()
        self._local = threading.local()
        self._thread: threading.Thread | None = None

    # -- recording -------------------------------------------------------------- #
    def add(self, name: str, seconds: float, count: int = 1, own_thread: bool = True) -> None:
        """Record ``seconds`` of stage ``name``. ``own_thread``: the seconds were spent on this thread (they count
        against its loop's untracked time); False for time measured elsewhere (another thread, another host)."""
        if not self.enabled:
            return
        seconds = max(float(seconds), 0.0)
        if own_thread and getattr(self._local, "depth", 0) == 0:
            self._local.tracked = getattr(self._local, "tracked", 0.0) + seconds
        ident = threading.get_ident()
        with self._lock:
            entry = self._stats.get(name)
            if entry is None:
                entry = self._stats[name] = [0, 0.0, 0.0, set()]
            entry[0] += count
            entry[1] += seconds
            entry[2] = max(entry[2], seconds / max(count, 1))
            entry[3].add(ident)
        self._ensure_reporter()

    @contextlib.contextmanager
    def time(self, name: str):
        """Record the ``with`` body's wall time as stage ``name``."""
        if not self.enabled:
            yield
            return
        local = self._local
        local.depth = getattr(local, "depth", 0) + 1
        t0 = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - t0
            local.depth -= 1
            self.add(name, seconds)

    @contextlib.contextmanager
    def loop_iteration(self, loop: str):
        """One pass of a thread's loop: what its own stages do not cover is recorded as ``<loop>.untracked``."""
        if not self.enabled:
            yield
            return
        local = self._local
        before = getattr(local, "tracked", 0.0)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - t0
            untracked = seconds - (getattr(local, "tracked", 0.0) - before)
            if untracked > 0:
                self.add(f"{loop}.untracked", untracked, own_thread=False)
            local.tracked = before + seconds

    # -- reporting -------------------------------------------------------------- #
    def _ensure_reporter(self) -> None:
        if self._thread is not None:
            return
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._report_loop, name="overhead-report", daemon=True)
            self._thread.start()

    def _report_loop(self) -> None:
        while True:
            time.sleep(self.period)
            with contextlib.suppress(Exception):
                text = self.report()
                if text:
                    logger.info(text)

    def take(self) -> tuple[float, dict[str, list]]:
        """The window's wall seconds and stats, starting a new window."""
        with self._lock:
            now = time.perf_counter()
            wall, self._t0 = now - self._t0, now
            stats, self._stats = self._stats, {}
        return wall, stats

    def report(self) -> str | None:
        """The table of the window since the last one (None when nothing was recorded), starting a new window."""
        wall, stats = self.take()
        if not stats or wall <= 0:
            return None
        lines = [
            f"[overhead] last {wall:.1f}s: stage, count, total s, % of wall, mean ms, max ms, threads "
            f"(compute = device or waiting on its output; wait = queued on something else)"
        ]
        for group in ("overhead", "wait", "compute", "idle"):
            names = sorted((n for n in stats if kind(n) == group), key=lambda n: -stats[n][1])
            if not names:
                continue
            # A stage's parts ("sched.admit.render" of "sched.admit") are listed, not added to the total again.
            total = sum(stats[n][1] for n in names if not any(n.startswith(p + ".") for p in stats))
            lines.append(f"  -- {group}: {total:.3f}s ({100 * total / wall:.1f}% of wall over all threads)")
            for name in names:
                count, seconds, peak, threads = stats[name]
                lines.append(
                    f"  {name:<34} {count:>7} {seconds:>9.3f} {100 * seconds / wall:>7.1f}% "
                    f"{1e3 * seconds / max(count, 1):>9.2f} {1e3 * peak:>9.2f} {len(threads):>3}"
                )
        return "\n".join(lines)


OVERHEAD = Overhead(
    enabled=os.environ.get("DEEPSEEK_OVERHEAD", "1") not in ("0", "", "false", "no"),
    period=float(os.environ.get("DEEPSEEK_OVERHEAD_PERIOD", "10")),
)
