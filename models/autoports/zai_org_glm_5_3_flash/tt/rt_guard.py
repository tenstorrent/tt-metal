# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14b: a guard against the generic_op program-cache hazard (2026-10-19 23:35Z chip-3 hang).

ttnn.generic_op caches a program by its descriptor WITHOUT the runtime args. A later launch whose descriptor matches
but whose kernels carry MORE runtime args per core reuses the cached program: the extra args are never written (the
host only logs `Index N is larger than runtime args size M`) and the kernel reads stale words -> hang. `checked(desc)`
remembers, per descriptor signature (kernel sources, compile-time args, defines, core ranges, CB / semaphore layout),
the runtime-arg count of every kernel on every core and raises BEFORE the enqueue when a later launch with the same
signature differs. Host-only, a few us per launch (launches inside a trace replay never come back here)."""

from __future__ import annotations

_SEEN: dict = {}


def _cores(crs) -> list:
    return sorted((x, y) for r in crs.ranges() for x in range(int(r.start.x), int(r.end.x) + 1)
                  for y in range(int(r.start.y), int(r.end.y) + 1))  # fmt: skip


def _signature(desc) -> tuple:
    ks = tuple((str(kd.kernel_source), tuple(int(a) for a in kd.compile_time_args),
                tuple(sorted((str(a), str(b)) for a, b in kd.defines)), str(kd.core_ranges)) for kd in desc.kernels)  # fmt: skip
    cbs = tuple((int(c.total_size), str(c.core_ranges)) for c in desc.cbs)
    sems = tuple((int(s.id), str(s.core_ranges)) for s in desc.semaphores)
    return ks, cbs, sems


def _len(kd, x, y) -> int:
    try:
        return len(kd.runtime_args[x][y])
    except IndexError:  # a core of the kernel's range without runtime args
        return 0


def _lengths(desc) -> tuple:
    return tuple(tuple(_len(kd, x, y) for x, y in _cores(kd.core_ranges)) for kd in desc.kernels)


def checked(desc):
    """desc (a ProgramDescriptor) unchanged; raises if a cached program with this signature had other arg counts"""
    sig, lens = _signature(desc), _lengths(desc)
    old = _SEEN.setdefault(sig, lens)
    if old != lens:
        bad = [i for i, (a, b) in enumerate(zip(old, lens)) if a != b]
        raise RuntimeError(f"rt_guard: generic_op would reuse a cached program with other runtime-arg counts (kernels "
                           f"{[str(desc.kernels[i].kernel_source).split('/')[-1] for i in bad]}): pad the args to a "
                           f"fixed length or put the count in a define")  # fmt: skip
    return desc
