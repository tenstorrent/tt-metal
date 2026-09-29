# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Grep-able timing events for device tests: one line per event, ``TT_EVENT {json}``, with a wall-clock UTC
timestamp, so a run's phases can be reconstructed from its log even though pytest prints captured output after
each test (not chronologically). Events: ``device.open`` / ``device.close`` (device fixtures),
``phase.begin`` / ``phase.end`` (test bodies: reference, oracle, weights, compute), ``cache.hit`` /
``cache.miss`` (result and weight caches, with key and bytes). Printing only: no behavior change for callers.
"""

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone

PREFIX = "TT_EVENT "


def emit(event: str, **fields) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event, **fields}
    print(PREFIX + json.dumps(record, default=str), flush=True)


@contextmanager
def phase(kind: str, **fields):
    """``phase.begin`` / ``phase.end`` (with seconds) of a ``kind`` phase (reference / oracle / weights / compute)
    around a block; ``phase.end`` is emitted on failure too."""
    emit("phase.begin", phase=kind, **fields)
    start = time.perf_counter()
    try:
        yield
    finally:
        emit("phase.end", phase=kind, seconds=round(time.perf_counter() - start, 3), **fields)


def cache(hit: bool, cache_name: str, key: str, nbytes: int | None = None) -> None:
    emit("cache.hit" if hit else "cache.miss", cache=cache_name, key=key, bytes=nbytes)
