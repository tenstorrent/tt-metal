# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stall watchdog for a blocking warmup step that Python cannot interrupt.

A device op stuck in C++ cannot be cancelled from Python, so a stalled step ends the process (the
worker supervisor restarts it, now with every kernel compiled so far in the on-disk cache). A slow
cold compile is not a stall: the JIT builds kernels in child processes, so their CPU time keeps rising
while it works. Only a step that makes no such progress for ``stall_s`` seconds counts as stuck.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
from typing import Callable

from loguru import logger

STALL_EXIT_CODE = 75  # EX_TEMPFAIL: the supervisor restarts the worker


def children_cpu_seconds() -> float:
    """CPU time of finished child processes (the JIT kernel builds)."""
    t = os.times()
    return t.children_user + t.children_system


def _exit_on_stall(name: str, stalled_s: float) -> None:
    logger.error(f"{name}: no progress for {stalled_s:.0f}s; dumping stacks and exiting {STALL_EXIT_CODE}")
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    sys.stderr.flush()
    os._exit(STALL_EXIT_CODE)


def run_with_stall_watchdog(
    fn: Callable[[], object],
    *,
    name: str,
    stall_s: float,
    poll_s: float = 10.0,
    retries: int = 1,
    progress: Callable[[], float] = children_cpu_seconds,
    on_stall: Callable[[str, float], None] = _exit_on_stall,
) -> object:
    """Run ``fn`` with a stall bound; an exception from ``fn`` is retried ``retries`` times in-process.

    ``stall_s <= 0`` disables the watchdog (``fn`` still gets its retries)."""
    for attempt in range(retries + 1):
        done = threading.Event()
        if stall_s > 0:

            def watch() -> None:
                last, last_t = progress(), time.monotonic()
                while not done.wait(poll_s):
                    now, now_t = progress(), time.monotonic()
                    if now > last:
                        last, last_t = now, now_t
                    elif now_t - last_t >= stall_s:
                        on_stall(name, now_t - last_t)
                        return

            threading.Thread(target=watch, name=f"watchdog:{name}", daemon=True).start()
        t0 = time.monotonic()
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - retried once, then re-raised
            if attempt >= retries:
                raise
            logger.warning(f"{name} failed after {time.monotonic() - t0:.0f}s ({e!r}); retrying once")
        finally:
            done.set()
    raise AssertionError("unreachable")
