# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from contextlib import contextmanager

# one stage per process, so module state is enough
_phases: list[tuple[str, float]] = []


def fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{secs:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{secs:04.1f}s"


@contextmanager
def phase(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _phases.append((name, elapsed))
        print(f"[time] {name}: {fmt(elapsed)}")


def record(name: str, seconds: float) -> None:
    _phases.append((name, seconds))
    print(f"[time] {name}: {fmt(seconds)}")


def summary(stage: str, total: float) -> None:
    print(f"\n[time] ─── {stage} ───")
    if not _phases:
        print(f"[time]   TOTAL  {fmt(total)}")
        return
    width = max(len(name) for name, _ in _phases)
    tracked = 0.0
    for name, elapsed in _phases:
        share = 100.0 * elapsed / total if total > 0 else 0.0
        print(f"[time]   {name:<{width}}  {fmt(elapsed):>10}  {share:5.1f}%")
        tracked += elapsed
    other = total - tracked
    if other > 0.05 * total and total > 0:
        print(f"[time]   {'(untimed)':<{width}}  {fmt(other):>10}  {100.0 * other / total:5.1f}%")
    print(f"[time]   {'TOTAL':<{width}}  {fmt(total):>10}")
