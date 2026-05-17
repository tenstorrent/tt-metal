# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN device helper.

Exposes a thin, defensive API on top of `ttnn` for:
  - opening / closing a mesh device safely (with cleanup on errors)
  - probing the actually-usable per-chip HBM by binary-searching tensor allocation
  - asking the device library for which mesh shapes it can open

This module is the only thing in tt_hw_planner that imports ttnn.  All
hardware-touching code is conditional on `ttnn` being importable; if it's
not, the calibrate / smoke-test commands print a helpful error and exit.
"""

from __future__ import annotations

import contextlib
from typing import Optional, Tuple


# ttnn is imported lazily so the module can be imported without tt-metal
# installed (the rest of the planner runs without it).
def _ttnn():
    try:
        import ttnn  # type: ignore

        return ttnn
    except ImportError as e:
        raise RuntimeError(
            "ttnn not available — required for `calibrate` and `smoke-test`.\n"
            "  Activate the tt-metal python env (`source python_env/bin/activate`)\n"
            "  and re-run.  This module is import-safe without ttnn for `plan`."
        ) from e


def _torch():
    import torch

    return torch


@contextlib.contextmanager
def open_mesh(shape: Tuple[int, int]):
    """Open a TTNN mesh of the requested shape (compute-only)."""
    ttnn = _ttnn()
    rows, cols = shape
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    try:
        yield mesh
    finally:
        try:
            ttnn.close_mesh_device(mesh)
        except Exception:
            pass


def usable_bytes_per_chip(
    mesh,
    dtype: str = "bfloat16",
    probe_layout: str = "TILE",
    min_gb: float = 0.5,
    max_gb: float = 64.0,
    tolerance_gb: float = 0.5,
) -> float:
    """
    Binary-search the largest single-tensor allocation that succeeds on the
    mesh.  This gives an HONEST measurement of the per-chip memory available
    after dispatch/CCL setup — which is exactly the constant our overhead
    model is trying to predict.

    Note: this allocates ONE tensor of size `N` GB; the true usable budget
    is slightly larger than the largest allocatable single tensor (allocator
    fragmentation makes the "largest contiguous" smaller than "total free").
    We treat this measurement as a LOWER BOUND on usable_per_chip.
    """
    ttnn = _ttnn()
    torch = _torch()

    dtype_map = {
        "bfloat16": (torch.bfloat16, ttnn.bfloat16, 2),
        "float32": (torch.float32, ttnn.float32, 4),
    }
    torch_dt, ttnn_dt, bytes_per_elem = dtype_map[dtype]
    layout = ttnn.TILE_LAYOUT if probe_layout == "TILE" else ttnn.ROW_MAJOR_LAYOUT

    def try_alloc(size_gb: float) -> bool:
        # Tile-aligned 2D tensor.  We use a square-ish shape rounded to 32.
        bytes_total = int(size_gb * 1e9)
        elems = bytes_total // bytes_per_elem
        side = int(elems**0.5)
        side = (side // 32) * 32
        if side <= 0:
            return False
        shape = (1, 1, side, side)
        try:
            t_cpu = torch.zeros(shape, dtype=torch_dt)
            t_dev = ttnn.from_torch(
                t_cpu, device=mesh, layout=layout, dtype=ttnn_dt, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(t_dev)
            return True
        except Exception:
            return False

    lo, hi = min_gb, max_gb
    last_success = 0.0
    while hi - lo > tolerance_gb:
        mid = (lo + hi) / 2
        if try_alloc(mid):
            last_success = mid
            lo = mid
        else:
            hi = mid
    return last_success


def mesh_info(mesh) -> dict:
    """Best-effort description of the open mesh (number of devices, etc.)."""
    ttnn = _ttnn()
    info = {}
    try:
        info["num_devices"] = mesh.get_num_devices()
    except Exception:
        pass
    try:
        info["shape"] = tuple(mesh.shape)
    except Exception:
        pass
    return info


def safe_quick_op(op_name: str, fn) -> dict:
    """Run a single TTNN op inside a try/except and capture timing/error info."""
    import time

    t0 = time.monotonic()
    try:
        fn()
        return {"op": op_name, "ok": True, "elapsed_s": time.monotonic() - t0, "error": None}
    except Exception as e:
        return {
            "op": op_name,
            "ok": False,
            "elapsed_s": time.monotonic() - t0,
            "error": f"{type(e).__name__}: {str(e).splitlines()[0]}",
        }
