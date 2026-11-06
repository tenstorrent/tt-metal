# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Engine start timing, off unless ``MIMO_STARTUP_TIMING=1``.

``stage(name, mesh)`` logs one ``MIMO_STARTUP <name> <seconds> s programs +<n> (total <m>)`` line per stage: the wall
time of the block with the device synchronized at both ends, and how many programs the mesh's program cache gained
(each one built: kernels looked up in the JIT cache or compiled, binaries loaded, the program object made). With
``MIMO_STARTUP_TIMING=2`` a stage given ``rerun`` runs its block a second time (every program now cached) and logs
that time too, which splits the first run into building programs (first - rerun) and running them (rerun)."""

import contextlib
import os
import time

from loguru import logger

import ttnn

LEVEL = int(os.environ.get("MIMO_STARTUP_TIMING", "0") or 0)
ENABLED = LEVEL > 0


def _programs(mesh):
    try:
        return mesh.num_program_cache_entries()
    except Exception:
        return -1


@contextlib.contextmanager
def stage(name, mesh=None):
    if not ENABLED:
        yield
        return
    if mesh is not None:
        ttnn.synchronize_device(mesh)
    p0 = _programs(mesh) if mesh is not None else 0
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if mesh is not None:
            ttnn.synchronize_device(mesh)
        dt = time.perf_counter() - t0
        p1 = _programs(mesh) if mesh is not None else 0
        logger.info(f"MIMO_STARTUP {name} {dt:.3f} s programs +{p1 - p0} (total {p1}) t={time.time():.3f}")


def rerun(name, mesh, fn):
    """With MIMO_STARTUP_TIMING=2: run ``fn`` again (all its programs cached) and log its time."""
    if LEVEL < 2:
        return
    with stage(f"{name}#rerun", mesh):
        fn()


def mark(name):
    if ENABLED:
        logger.info(f"MIMO_STARTUP_MARK {name} t={time.time():.3f}")
