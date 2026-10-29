# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Core range merging for the fused programs' descriptors (``QWEN38_RANGE_MERGE=1``, default off).

Program dispatch costs ~0.4 us per CoreRange of EACH CB descriptor, kernel and semaphore, and a CoreRangeSet built from single
cores stays one range per core.  ``merge`` rewrites any CoreRangeSet as maximal rectangles (the cores' rows merged into spans,
equal spans of adjacent rows stacked): the same cores, so the program is the same; runtime args stay per core.  ``program.py``
applies it in ``cb_descriptor`` / ``semaphore_descriptor`` / ``_kernel`` and (``merge_program``) to plain ``ProgramDescriptor``s
at ``run_program``.  With ``QWEN38_RANGE_STATS=1`` every descriptor's range count before and after is recorded per program and
printed at exit (``RANGE_STATS``), whether or not the merge is on.
"""

from __future__ import annotations

import atexit
import os
from collections import defaultdict

import ttnn

ENV = "QWEN38_RANGE_MERGE"
STATS_ENV = "QWEN38_RANGE_STATS"
_PENDING: list[tuple[str, int, int]] = []
PROGRAMS: dict[str, dict] = defaultdict(
    lambda: {"launches": 0, "before": 0, "after": 0, "kinds": defaultdict(lambda: [0, 0])}
)


def enabled(environ=None) -> bool:
    return (os.environ if environ is None else environ).get(ENV, "0") == "1"


def stats_enabled(environ=None) -> bool:
    return (os.environ if environ is None else environ).get(STATS_ENV, "0") == "1"


def cores_of(core_range_set) -> set[tuple[int, int]]:
    cores = set()
    for r in core_range_set.ranges():
        for x in range(r.start.x, r.end.x + 1):
            for y in range(r.start.y, r.end.y + 1):
                cores.add((x, y))
    return cores


def rectangles(cores: set[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """Maximal-ish rectangles (x0, y0, x1, y1) covering exactly ``cores``: per row the consecutive x spans, then equal spans of
    consecutive rows stacked."""

    spans: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for y in sorted({y for _, y in cores}):
        xs = sorted(x for x, yy in cores if yy == y)
        start = prev = xs[0]
        for x in xs[1:]:
            if x != prev + 1:
                spans[y].append((start, prev))
                start = x
            prev = x
        spans[y].append((start, prev))
    open_rects: dict[tuple[int, int], list[int]] = {}  # (x0, x1) -> [y0, y1] still growing
    done: list[tuple[int, int, int, int]] = []
    last_y = None
    for y in sorted(spans):
        current = set(spans[y])
        if last_y is not None and y != last_y + 1:  # a gap of rows closes everything
            for key, (y0, y1) in open_rects.items():
                done.append((key[0], y0, key[1], y1))
            open_rects = {}
        for key in list(open_rects):
            if key not in current:
                y0, y1 = open_rects.pop(key)
                done.append((key[0], y0, key[1], y1))
        for key in current:
            if key in open_rects:
                open_rects[key][1] = y
            else:
                open_rects[key] = [y, y]
        last_y = y
    for key, (y0, y1) in open_rects.items():
        done.append((key[0], y0, key[1], y1))
    return sorted(done, key=lambda r: (r[1], r[0]))


def merge(core_range_set):
    """``core_range_set`` as maximal rectangles (the same cores)."""

    cores = cores_of(core_range_set)
    if not cores:
        return core_range_set
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1 in rectangles(cores)]
    )


def apply(core_range_set, kind: str):
    """The descriptor builders' entry: the merged set when the merge is on, else the set itself; counts when stats are on."""

    if not enabled() and not stats_enabled():
        return core_range_set
    merged = merge(core_range_set)
    if stats_enabled():
        _PENDING.append((kind, len(core_range_set.ranges()), len(merged.ranges())))
    return merged if enabled() else core_range_set


def merge_program(descriptor) -> None:
    """In-place merge of a plain ProgramDescriptor's kernels / CBs / semaphores (descriptors built outside ``program.py``)."""

    if not enabled() and not stats_enabled():
        return
    for kind, items in (("kernel", descriptor.kernels), ("cb", descriptor.cbs), ("semaphore", descriptor.semaphores)):
        for item in items:
            if kind == "cb":
                try:
                    if item.has_buffer():  # a CB tied to a sharded buffer keeps its ranges
                        continue
                except TypeError:
                    if item.has_buffer:
                        continue
            merged = merge(item.core_ranges)
            if stats_enabled():
                _PENDING.append((kind + "*", len(item.core_ranges.ranges()), len(merged.ranges())))
            if enabled():
                item.core_ranges = merged


def flush(tag: str) -> None:
    """At a launch: the descriptors built since the last launch belong to ``tag`` (a mesh program builds one per device)."""

    if not stats_enabled() or not _PENDING:
        return
    entry = PROGRAMS[tag]
    entry["launches"] += 1
    for kind, before, after in _PENDING:
        entry["before"] += before
        entry["after"] += after
        entry["kinds"][kind][0] += before
        entry["kinds"][kind][1] += after
    _PENDING.clear()


def _report() -> None:
    if not PROGRAMS:
        return
    lines = [
        "RANGE_STATS tag launches ranges_before ranges_after (summed over the launch's descriptors; a mesh launch builds one per device)"
    ]
    for tag, e in sorted(PROGRAMS.items(), key=lambda kv: -kv[1]["before"] / max(kv[1]["launches"], 1)):
        n = max(e["launches"], 1)
        kinds = {k: (round(b / n, 1), round(a / n, 1)) for k, (b, a) in e["kinds"].items()}
        lines.append(f"RANGE_STATS {tag} {e['launches']} {e['before'] / n:.1f} {e['after'] / n:.1f} {kinds}")
    print("\n".join(lines), flush=True)


atexit.register(_report)
