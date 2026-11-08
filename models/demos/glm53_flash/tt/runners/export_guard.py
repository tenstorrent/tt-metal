# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The socket guard of the GLM-5.3 Flash prefill runner's async export (``GLM53F_EXPORT_ASYNC``, tt-blaze G53F-0309).

With the async export, a worker thread packs + writes chunk i's KV and then sends its layer acks while the main thread
already prefills chunk i+1. After the LAST chunk of a prompt the main thread goes back to the H2D socket and waits for
a chunk that the engine only sends once it holds every ack of this prompt -- and the worker cannot finish while the
main thread sits in that wait: the wait is ``_decode_metadata`` -> ``ttnn.to_torch`` -> ``ttnn.from_device``, whose
binding does not release the GIL, so the worker blocks in take_gil (tt-blaze G53F-0256 22:14 hang; mechanism MEASURED
with gdb + faulthandler in G53F-0309). The engine's 12-byte chunk header (slot, start, end) does not say which chunk is
the last, and a FULL chunk can be the last one (a prompt that ends on a chunk boundary, a declined burst, the max-seq
workaround).

``ExportDrainGuard.before_socket_wait`` runs just before the main thread blocks on the socket. While export jobs are
pending it polls ``probe`` (a non-blocking look at the socket: the data-ready semaphore and the inline header of the
next transfer, read raw over the NOC), and returns at once when a real chunk is already waiting -- the next socket call
then returns without blocking, and the export keeps overlapping that chunk's prefill. Otherwise it waits, polling,
until the worker has drained the queue (acks out), so the main thread never blocks on the engine with acks pending.
A sentinel (shutdown / warm-up: header words < 0) also drains first. No probe (or a probe that fails) means a plain
drain before every socket wait: no overlap, never a hang.

Pure Python (no ttnn import), so the policy is unit-tested on the host: ``tests/test_export_guard.py``.
"""

from __future__ import annotations

import queue
import time
from typing import Callable, Optional, Sequence

# probe() -> None when no transfer is waiting in the socket, else the waiting transfer's header words
Probe = Callable[[], Optional[Sequence[int]]]


class ExportDrainGuard:
    def __init__(
        self,
        jobs: queue.Queue,
        probe: Optional[Probe] = None,
        *,
        poll_s: float = 0.0005,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        log: Optional[Callable[[str], None]] = None,
    ):
        self.jobs = jobs
        self.probe = probe
        self.poll_s = poll_s
        self.clock = clock
        self.sleep = sleep
        self.log = log or (lambda msg: None)
        self.overlapped = 0  # socket waits entered with export pending because the next chunk was already there
        self.drained = 0  # socket waits that first drained the export queue (a prompt's last chunk, or a sentinel)
        self.probe_error: Optional[str] = None

    def _peek(self) -> Optional[Sequence[int]]:
        if self.probe is None:
            return None
        try:
            return self.probe()
        except Exception as e:  # noqa: BLE001 -- a failed probe degrades to drain-always, never to a hang
            self.probe_error = f"{type(e).__name__}: {e}"
            self.log(f"[glm53f-export] socket probe failed ({self.probe_error}); draining before every socket wait")
            self.probe = None
            return None

    def before_socket_wait(self) -> str:
        """``"idle"`` (nothing pending), ``"overlap"`` (a real chunk is waiting: the export runs on beside it) or
        ``"drained"`` (the export queue was emptied first)."""
        if not self.jobs.unfinished_tasks:
            return "idle"
        t0, pending, waited_on = self.clock(), self.jobs.unfinished_tasks, None
        while self.jobs.unfinished_tasks:
            head = self._peek()
            if head is not None:
                if min(int(w) for w in head) >= 0:
                    self.overlapped += 1
                    return "overlap"
                waited_on = tuple(int(w) for w in head)  # a sentinel: drain before the runner consumes it
            self.sleep(self.poll_s)
        self.drained += 1
        why = "no chunk waiting" if waited_on is None else f"a sentinel {waited_on} waiting"
        self.log(
            f"[glm53f-export] socket guard: {why}, drained {pending} export job(s) in "
            f"{(self.clock() - t0) * 1e3:.1f} ms before the socket wait (overlapped {self.overlapped}, "
            f"drained {self.drained} so far{'' if self.probe is not None else '; NO probe: drain-always'})"
        )
        return "drained"


__all__ = ["ExportDrainGuard", "Probe"]
