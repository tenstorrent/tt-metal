# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Timer-driven heartbeat so long device/compute phases are never silent.

Per-item progress only ticks between items; a single op that is CPU-starved or hung
shows nothing. ``Watchdog`` logs from a daemon thread on a fixed interval, independent of
progress, so any stall surfaces within ``interval`` seconds instead of reading as a hang.
Best-effort: a C-extension stretch that never releases the GIL can still block the thread.
"""

from __future__ import annotations

import threading
import time

from loguru import logger


class Watchdog:
    """Context manager that logs ``<label>: still working, Ns elapsed`` every ``interval``
    seconds until the wrapped block exits, then logs the total."""

    def __init__(self, label: str, interval: float = 20.0, *, log_done: bool = True) -> None:
        self._label = label
        self._interval = interval
        self._log_done = log_done
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tt-watchdog", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            logger.info(f"{self._label}: still working, {time.monotonic() - self._t0:.0f}s elapsed")

    def __enter__(self) -> Watchdog:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self._log_done:
            logger.info(f"{self._label}: done in {time.monotonic() - self._t0:.0f}s")
