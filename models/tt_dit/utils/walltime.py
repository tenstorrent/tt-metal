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
import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field

_LOCK = threading.RLock()
# Wall baseline for entrypoints (``python -m ...``) that have no external duration
# to hand us; pytest passes the per-item call duration instead (see conftest).
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


# ``_current`` is reset between pytest items; ``_session`` accumulates for the whole
# process (and a multi-item session rollup). Outside pytest the two move together.
_current = _Ledger()
_session = _Ledger()


def _enabled() -> bool:
    return os.environ.get("TT_WALLTIME", "1") != "0"


def record(
    category: str, label: str, seconds: float, *, cached: bool | None = None, count: int = 1, detail: str = ""
) -> None:
    """Accumulate ``seconds`` into ``category`` for both the per-item and session ledgers.
    ``cached`` tallies HIT (True) / MISS (False); a MISS is also retained for the anomaly list."""
    if not _enabled():
        return
    with _LOCK:
        for led in (_current, _session):
            cat = led.cats.get(category)
            if cat is None:
                cat = _Cat()
                led.cats[category] = cat
            cat.seconds += seconds
            cat.count += count
            if cached is True:
                cat.hits += 1
            elif cached is False:
                cat.misses += 1
                led.misses.append((label, seconds, detail))


@contextmanager
def timed(category: str, label: str, *, cached: bool | None = None):
    t0 = time.monotonic()
    try:
        yield
    finally:
        record(category, label, time.monotonic() - t0, cached=cached)


def reset() -> None:
    """Clear the per-item ledger (the session ledger keeps accumulating)."""
    global _current
    with _LOCK:
        _current = _Ledger()


def render(title: str, ledger: _Ledger | None = None, wall: float | None = None) -> str:
    """Render a ledger as a table reconciled to wall time.

    ``wall`` is the true end-to-end wall for the scope (pytest call duration, or
    process time since import when omitted). The table lists each category's
    seconds and share of wall, then an ``untracked`` remainder (wall − tracked)
    and ``TOTAL (wall)`` so the breakdown always reconciles. Anomalies list every
    cached=False weight load.
    """
    led = ledger if ledger is not None else _current
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


def render_session(title: str, wall: float | None = None) -> str:
    """Render the cumulative session ledger (every item since process start)."""
    return render(title, _session, wall)


def _atexit() -> None:
    # pytest gets its per-item/session blocks from conftest hooks; only print here for the
    # `python -m ...` entrypoints that have no such hook. PYTEST_CURRENT_TEST is cleared by
    # session end (before this atexit runs), so detect pytest by its imported module instead.
    if not _enabled() or "pytest" in sys.modules:
        return
    with _LOCK:
        nonempty = bool(_session.cats)
    if nonempty:
        print(render("end of run", _session))  # noqa: T201


atexit.register(_atexit)
