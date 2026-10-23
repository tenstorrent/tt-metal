# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Process-global wall-time ledger with cache-miss anomaly flags.

A run's wall time hides in a few cost centers (weight conversion/load, warmup
compile, generation, audio decode). When something that should be cached isn't —
e.g. TT_DIT_CACHE_DIR unset — the run is silently slow. This ledger collects
per-category time and, at the end of a run, prints where the time went, an
``untracked`` remainder that reconciles the table to true wall time (so the
breakdown can't mislead), and an ANOMALIES section so "should've been fast,
wasn't" is obvious instead of buried in thousands of log lines.

Fed from existing cost centers (no scattered timers): ``utils.cache.load_model``
(weight loads, with HIT/MISS), ``utils.progress.Watchdog`` (phases), and a few
dominant pipeline phases wrapped at their existing timing sites. Disable with
TT_WALLTIME=0.

NOT covered: JIT kernel-build-cache hit rate. That counter ("JIT cache stats:
H/T hits") is emitted by a C++ static destructor at process teardown — after
Python has exited — and ttnn exposes no pre-teardown Python accessor for it, so
it cannot be folded into this in-process ledger. Read it from the device log.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field

_LOCK = threading.RLock()
# Wall baseline: render() falls back to (now - import) when no explicit wall is handed in.
_T_IMPORT = time.monotonic()


@dataclass
class _Cat:
    seconds: float = 0.0
    count: int = 0
    hits: int = 0  # cached is True
    misses: int = 0  # cached is False


@dataclass
class _Ledger:
    cats: "OrderedDict[str, _Cat]" = field(default_factory=OrderedDict)
    # (label, seconds, detail) for every cached=False record, surfaced as anomalies.
    misses: list = field(default_factory=list)


# One process-global ledger, rendered once at teardown (see ``_atexit``).
_ledger = _Ledger()


# Read once at import: an instrumentation toggle should hold for a whole process, and record()
# consults it on every span.
_ENABLED = os.environ.get("TT_WALLTIME", "1") != "0"


def _enabled() -> bool:
    return _ENABLED


def record(
    category: str, label: str, seconds: float, *, cached: bool | None = None, count: int = 1, detail: str = ""
) -> None:
    """Accumulate ``seconds`` into ``category``. ``cached`` tallies HIT (True) / MISS (False); a MISS
    is also retained for the anomaly list."""
    if not _enabled():
        return
    with _LOCK:
        cat = _ledger.cats.get(category)
        if cat is None:
            cat = _Cat()
            _ledger.cats[category] = cat
        cat.seconds += seconds
        cat.count += count
        if cached is True:
            cat.hits += 1
        elif cached is False:
            cat.misses += 1
            _ledger.misses.append((label, seconds, detail))


@contextmanager
def timed(category: str, label: str, *, cached: bool | None = None):
    t0 = time.monotonic()
    try:
        yield
    finally:
        record(category, label, time.monotonic() - t0, cached=cached)


def render(title: str, ledger: _Ledger | None = None, wall: float | None = None) -> str:
    """Render a ledger as a table reconciled to wall time.

    ``wall`` is the true end-to-end wall for the scope (an explicit duration, or
    process time since import when omitted). The table lists each category's
    seconds and share of wall, then an ``untracked`` remainder (wall − tracked)
    and ``TOTAL (wall)`` so the breakdown always reconciles. Anomalies list every
    cached=False weight load. Categories are assumed disjoint; a phase span that
    nests a weight_load would double-count and clamp ``untracked`` to 0.
    """
    led = ledger if ledger is not None else _ledger
    with _LOCK:
        cats = list(led.cats.items())
        misses = list(led.misses)

    tracked = sum(c.seconds for _, c in cats)
    if wall is None:
        wall = time.monotonic() - _T_IMPORT
    wall = max(wall, tracked)  # never report a wall smaller than what we measured
    untracked = wall - tracked
    denom = wall or 1.0

    width = 74
    lines = ["", "=" * width, f"WALL-TIME LEDGER · {title}", "-" * width]
    lines.append(f"{'category':<18}{'seconds':>10}{'%wall':>8}{'count':>8}   notes")
    for name, c in cats:
        note = f"{c.hits} HIT / {c.misses} MISS" if (c.hits or c.misses) else ""
        lines.append(f"{name:<18}{c.seconds:>10.1f}{100.0 * c.seconds / denom:>7.1f}%{c.count:>8}   {note}")
    lines.append(f"{'untracked':<18}{untracked:>10.1f}{100.0 * untracked / denom:>7.1f}%")
    lines.append("-" * width)
    lines.append(f"{'TOTAL (tracked)':<18}{tracked:>10.1f}{100.0 * tracked / denom:>7.1f}%")
    lines.append(f"{'TOTAL (wall)':<18}{wall:>10.1f}{100.0:>7.1f}%")

    anomalies = [
        f"CACHE MISS: weight_load {label} ({seconds:.1f}s) — {detail or 'TT_DIT_CACHE_DIR unset or blocking key changed'}"
        for label, seconds, detail in misses
    ]
    lines.append("")
    if anomalies:
        lines.append("ANOMALIES:")
        lines.extend(f"  - {a}" for a in anomalies)
    else:
        lines.append("ANOMALIES: none")
    lines.append("=" * width)
    return "\n".join(lines)


def _atexit() -> None:
    # The ledger surfaces once, at process teardown — under pytest (the tests wire no per-item hook) and
    # the ``python -m ...`` entrypoints alike.
    if not _enabled():
        return
    with _LOCK:
        nonempty = bool(_ledger.cats)
    if not nonempty:
        return
    try:
        print(render("end of run", _ledger))  # noqa: T201
    except (ValueError, OSError):
        pass  # stdout may already be closed during interpreter shutdown


atexit.register(_atexit)
