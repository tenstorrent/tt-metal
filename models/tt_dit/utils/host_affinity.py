# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Pin the process to one hardware thread per physical core on SMT hosts.

Trace replay keeps the device busy, but the readback and dispatch threads that sit between the
stages (latent hand-offs, the VAE frame readback, the audio chain) are latency-critical host work.
On a host with SMT, two of those threads can land on sibling hardware threads of one core and run at
roughly half speed; on the 32-core/64-thread galaxy host restricting the affinity mask to one sibling
per core removes the sharing. A thread-count cap alone did nothing.

Two mechanisms are provided:

1. :func:`pin_one_thread_per_core` narrows the affinity mask of full-mask threads in the running
   process. This is a best-effort safety net: torch/OMP thread pools and tt-metal dispatch/reader
   threads are already spawned (at import and at device open) across both sibling sets before the LTX
   pipeline __init__ runs, so ``sched_setaffinity`` after the fact cannot migrate them. In-process
   pinning measured ~6.7-7.5 s on the galaxy ring traced replay.

2. :func:`reexec_pinned_before_torch` sets the affinity mask and ``os.execv`` re-execs the python
   process BEFORE torch is imported. The re-execed process inherits the pinned mask from PID start, so
   torch/OMP pools and tt-metal device threads land inside the chosen cores -- equivalent to launching
   under ``taskset`` but fully in-process. This is the mechanism that reaches the taskset-class
   numbers: galaxy ring traced replay 6.2 s re-execed (measured, no taskset) vs 6.6-6.7 s without.
   The LTX test conftest calls it from the earliest pytest hook by default (``LTX_PIN_PREIMPORT=0``
   skips it); any other entry point (a server, a script) should call it as its first statement, before
   importing torch or ttnn and before any device is opened (re-exec after device open risks a wedge).

``LTX_PIN_CORES=0`` disables both; on hosts without SMT (or without the sysfs topology) both are no-ops.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

from loguru import logger

_SYSFS_CPU = "/sys/devices/system/cpu"
_applied: set[int] | None = None


def _chosen_cores() -> set[int] | None:
    """The core set the pin targets: one hardware thread per physical core within the current mask.

    Pure host topology read (sysfs); does NOT import torch/ttnn and does NOT open the device.
    Returns ``None`` when there is nothing to narrow (no SMT / no topology / mask already narrow).
    """
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        current = set(os.sched_getaffinity(0))
    except OSError:
        return None
    chosen = one_thread_per_core(allowed=current)
    if not chosen:
        return None
    chosen_set = set(chosen)
    if chosen_set == current:
        return None
    return chosen_set


def reexec_pinned_before_torch(reason: str = "LTX pipeline (pre-import re-exec)") -> None:
    """Pin to the chosen cores and re-exec this python process so torch/ttnn inherit the mask.

    Must be called as early as possible (before torch/ttnn import and before any device is opened).
    ``os.execv`` replaces the current process image; the re-execed process inherits the affinity mask
    from PID start, so torch/OMP thread pools and tt-metal dispatch/reader threads all land inside the
    chosen cores -- the taskset-equivalent placement, fully in-process.

    No-op (returns without re-exec) when:
      * ``LTX_PIN_CORES=0`` (pinning globally disabled), or
      * ``_LTX_REEXECED`` is already set (we are the re-execed child -- prevents an infinite loop), or
      * the current affinity mask already equals the chosen core set (nothing to do), or
      * the host has no SMT / no readable topology (``_chosen_cores`` returns ``None``).

    Importable and callable without torch/ttnn present; it never imports them and never opens a device.
    """
    if os.environ.get("LTX_PIN_CORES", "1") in ("0", "false", "False"):
        return
    if os.environ.get("_LTX_REEXECED"):
        return
    if not hasattr(os, "sched_getaffinity"):
        return
    chosen = _chosen_cores()
    if not chosen:
        return
    try:
        current = set(os.sched_getaffinity(0))
    except OSError:
        return
    if current == chosen:
        return
    try:
        os.sched_setaffinity(0, chosen)
    except OSError:
        return
    os.environ["_LTX_REEXECED"] = "1"
    logger.info(
        f"host affinity: {reason}: re-exec pinned to {len(chosen)} cores {sorted(chosen)} "
        f"(was {len(current)} CPUs); torch/ttnn will inherit this mask (LTX_PIN_CORES=0 disables)"
    )
    os.execv(sys.executable, [sys.executable] + sys.argv)


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
    """Narrow every thread that still carries the full CPU mask to one hardware thread per core.

    Only threads whose mask is the whole online set are touched: tt-metal pins its per-device dispatch
    and reader threads to single CPUs of its own choosing (spread over both sibling sets), and those
    placements must be left alone -- re-pinning them measured worse than not pinning at all. Threads
    created later inherit their creator's mask, and helpers born before the first call can still spawn
    full-mask threads, so the call is repeated after the device is open; it is cheap and idempotent per
    thread. torch's intra-op pool is capped to the chosen core count as well (it sizes itself from the
    CPU count it saw at import). Honours ``LTX_PIN_CORES=0``; no-op without SMT.
    """
    if os.environ.get("LTX_PIN_CORES", "1") in ("0", "false", "False"):
        return None
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        full = set(os.sched_getaffinity(0)) if _applied is None else _full_set()
    except OSError:
        return None
    chosen = _applied if _applied is not None else set(one_thread_per_core(allowed=full) or [])
    if not chosen or chosen == full:
        return None
    narrowed = 0
    for tid in _thread_ids():
        try:
            mask = set(os.sched_getaffinity(tid))
        except OSError:
            continue  # exited between listing and reading
        if mask != full:
            continue  # explicitly placed (tt-metal) or already narrowed: leave it
        try:
            os.sched_setaffinity(tid, chosen)
            narrowed += 1
        except OSError:
            continue
    first = _applied is None
    _set_applied(chosen, full)
    if first:
        _cap_torch_threads(len(chosen))
    if narrowed or first:
        logger.info(
            f"host affinity: {reason}: {len(chosen)} of {len(full)} CPUs (one hardware thread per core), "
            f"narrowed {narrowed} full-mask threads (LTX_PIN_CORES=0 disables)"
        )
    return sorted(chosen)


_full: set[int] | None = None


def _full_set() -> set[int]:
    return set(_full) if _full is not None else set(os.sched_getaffinity(0))


def _set_applied(chosen: set[int], full: set[int]) -> None:
    global _applied, _full
    _applied, _full = set(chosen), set(full)


def _cap_torch_threads(n: int) -> None:
    """torch sizes its intra-op pool from the CPU count it saw at import; cap it to the pinned cores."""
    try:
        import torch

        if torch.get_num_threads() > n:
            torch.set_num_threads(n)
    except Exception:  # torch absent or pool already fixed: nothing to do
        return


def _thread_ids() -> list[int]:
    """All thread ids of this process (``/proc/self/task``); falls back to just the caller."""
    try:
        tids = sorted(int(t) for t in os.listdir("/proc/self/task"))
    except OSError:
        return [0]
    return tids or [0]
