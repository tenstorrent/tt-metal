# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Pin the process to one hardware thread per physical core on SMT hosts.

Trace replay keeps the device busy, but the readback and dispatch threads that sit between the
stages (latent hand-offs, the VAE frame readback, the audio chain) are latency-critical host work.
On a host with SMT, two of those threads can land on sibling hardware threads of one core and run at
roughly half speed; on the 32-core/64-thread galaxy host that cost ~0.5 s of a 6.2 s traced
generation (ring replay 6.6-7.1 s unpinned vs 6.1-6.3 s pinned, 4/4 runs; a thread-count cap alone
did nothing). Restricting the affinity mask to one sibling per core removes the sharing.

``LTX_PIN_CORES=0`` disables it; on hosts without SMT (or without the sysfs topology) it is a no-op.

Apply it BEFORE the mesh is opened: tt-metal places its dispatch and reader threads at device open from
the mask it sees then; re-pinning those threads afterwards measured worse than not pinning at all.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

from loguru import logger

_SYSFS_CPU = "/sys/devices/system/cpu"
_applied: set[int] | None = None


def _parse_cpu_list(text: str) -> list[int]:
    """Parse a sysfs cpu list like ``0,32`` or ``0-3,8-11`` into sorted ints."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            cpus.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def one_thread_per_core(sysfs_cpu: str | None = None, allowed: set[int] | None = None) -> list[int] | None:
    """Return one CPU per physical core (the lowest-numbered sibling), restricted to ``allowed``.

    Returns ``None`` when the topology is unavailable or the host has no SMT (every core has one
    sibling), so callers can leave the affinity mask alone.
    """
    sysfs_cpu = sysfs_cpu or _SYSFS_CPU
    sibling_files = glob.glob(os.path.join(sysfs_cpu, "cpu[0-9]*", "topology", "thread_siblings_list"))
    if not sibling_files:
        return None
    groups: dict[tuple[int, ...], None] = {}
    for f in sibling_files:
        try:
            siblings = tuple(_parse_cpu_list(Path(f).read_text()))
        except OSError:
            return None
        if siblings:
            groups[siblings] = None
    if not groups or all(len(g) == 1 for g in groups):
        return None
    chosen: list[int] = []
    for siblings in groups:
        candidates = [c for c in siblings if allowed is None or c in allowed]
        if candidates:
            chosen.append(min(candidates))
    return sorted(chosen) or None


def pin_one_thread_per_core(reason: str = "LTX pipeline") -> list[int] | None:
    """Apply the one-sibling-per-core affinity mask to this process (idempotent).

    Only narrows the current mask (never widens a mask an operator already set), honours
    ``LTX_PIN_CORES=0``, and logs once. Returns the CPUs pinned to, or ``None`` if nothing changed.
    """
    global _applied
    if os.environ.get("LTX_PIN_CORES", "1") in ("0", "false", "False"):
        return None
    if _applied is not None:
        return sorted(_applied)
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        current = set(os.sched_getaffinity(0))
    except OSError:
        return None
    chosen = one_thread_per_core(allowed=current)
    if chosen is None or set(chosen) == current:
        return None
    # Affinity is per thread on Linux, and the device's dispatch and reader threads already exist by the
    # time a pipeline is built (the mesh is opened first), so pin every thread of the process, not just
    # the caller; threads created later inherit their creator's mask.
    pinned, failed = 0, 0
    for tid in _thread_ids():
        try:
            os.sched_setaffinity(tid, set(chosen))
            pinned += 1
        except OSError:
            failed += 1  # a thread that exited between listing and pinning
    if pinned == 0:
        logger.warning("host affinity: could not pin any thread; leaving the mask alone")
        return None
    _applied = set(chosen)
    logger.info(
        f"host affinity: {reason} pinned to one hardware thread per core: {len(chosen)} of {len(current)} CPUs, "
        f"{pinned} threads (LTX_PIN_CORES=0 disables)"
    )
    return sorted(_applied)


def _thread_ids() -> list[int]:
    """All thread ids of this process (``/proc/self/task``), the calling thread first; falls back to just 0."""
    try:
        tids = sorted(int(t) for t in os.listdir("/proc/self/task"))
    except OSError:
        return [0]
    return tids or [0]
