# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.15 (agent rule 18, owner 2026-10-30): the Tensix-core cap of one op on a Galaxy (setting MAX_WORKER_CORES).

The Galaxy chassis' power supply trips when a chip's current rises too fast (di/dt); the full 12 x 10 = 120-core grid
can do it, so no op may run on more than MAX_WORKER_CORES (110) cores of a chip there. Every op of the model that
picks a core grid takes it from here: the matmul program configs (tt/mm_configs.py: ``worker_grid``), the bring-up
forks (their ``max_grid`` option: ``max_grid``), the stock ops with a ``sub_core_grids`` option (``sub_cores``).
Off a Galaxy (or MAX_WORKER_CORES=0) ``max_grid`` / ``sub_cores`` are None: every call is the one it was.

``init_caps(mesh)`` (model / module construction, tt/common.compute_config) records the mesh, so ``sub_cores()`` with no
argument answers for it; with a tensor or a mesh it answers for that device."""

from __future__ import annotations

import ttnn

from .settings import settings

_GALAXY = None


def is_galaxy() -> bool:
    """Whether the chips sit in a Galaxy chassis (agent rule 18). Read once per process from the cluster type."""
    global _GALAXY
    if _GALAXY is None:
        try:
            ct = ttnn.cluster.get_cluster_type()
            C = ttnn.cluster.ClusterType
            _GALAXY = ct in tuple(getattr(C, n) for n in ("GALAXY", "TG", "BLACKHOLE_GALAXY") if hasattr(C, n))
        except Exception:  # a box with no cluster type (e.g. 8 x P150) is not a Galaxy
            _GALAXY = False
    return _GALAXY


def max_worker_cores(mesh) -> int:
    """Setting MAX_WORKER_CORES (FIX.15, agent rule 18): the most Tensix cores of a chip one op may run on. The
    setting on a Galaxy, the full compute grid elsewhere (or with MAX_WORKER_CORES=0)."""
    g = mesh.compute_with_storage_grid_size()
    full = g.x * g.y
    cap = int(settings.get("MAX_WORKER_CORES"))
    return min(cap, full) if cap > 0 and is_galaxy() else full


def worker_grid(mesh) -> ttnn.CoreCoord:
    """The compute grid an op may use under max_worker_cores: the full grid's rows, as many columns as fit (12 x 10
    -> 11 x 10 under 110). A rectangle starting at (0, 0), so a program config's compute_with_storage_grid_size."""
    g = mesh.compute_with_storage_grid_size()
    cap = max_worker_cores(mesh)
    if g.x * g.y <= cap:
        return ttnn.CoreCoord(g.x, g.y)
    if cap >= g.y:
        return ttnn.CoreCoord(cap // g.y, g.y)
    return ttnn.CoreCoord(cap, 1)


def worker_cores(mesh) -> ttnn.CoreRangeSet:
    """worker_grid as a CoreRangeSet (an op's sub_core_grids / core_grid option)."""
    w = worker_grid(mesh)
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(w.x - 1, w.y - 1))})


_CAPS = {}  # mesh id -> (max_grid CoreCoord | None, sub_core_grids CoreRangeSet | None)


def _caps(dev) -> tuple:
    """(max_grid, sub_core_grids) of the model's mesh (init_caps: one mesh a process, a dict lookup a call), else of
    ``dev`` (a mesh or a tensor on one; a host tensor or None: no cap)."""
    hit = _CAPS.get(None)
    if hit is not None:
        return hit
    if dev is None:
        return (None, None)
    if not hasattr(dev, "compute_with_storage_grid_size"):
        dev = dev.device() if hasattr(dev, "device") else None
        if dev is None:
            return (None, None)
    key = dev.id()
    hit = _CAPS.get(key)
    if hit is None:
        g, w = dev.compute_with_storage_grid_size(), worker_grid(dev)
        capped = w.x * w.y < g.x * g.y
        hit = _CAPS[key] = (w, worker_cores(dev)) if capped else (None, None)
    return hit


def init_caps(mesh) -> None:
    """Record the mesh the model runs on (sub_cores() / max_grid() with no argument)."""
    _CAPS[None] = _caps(mesh)


def max_grid(dev=None):
    """The ``max_grid`` option of the bring-up forks (FIX.15, MAX_WORKER_CORES) for a mesh or a tensor on it: the
    capped grid, or None where no cap applies (the op's whole grid, the call as before)."""
    return _caps(dev)[0]


def sub_cores(dev=None):
    """The ``sub_core_grids`` of a stock op (FIX.15, MAX_WORKER_CORES) for a mesh or a tensor on it: the capped
    grid's cores, or None where no cap applies (the op picks its grid, the call as before)."""
    return _caps(dev)[1]


def to_rm(x: ttnn.Tensor, memory_config=None) -> ttnn.Tensor:
    """ttnn.to_layout(x, ROW_MAJOR) under the cap (FIX.15): stock untilize takes sub_core_grids only for a tensor one
    tile row tall (untilize_device_operation.cpp), so a capped mesh untilizes through ttnn.untilize_with_unpadding over
    the whole tensor (no unpadding; its interleaved factories split the rows over sub_core_grids): the same values.
    Without a cap: ttnn.to_layout, the call as before."""
    sc = sub_cores(x)
    if sc is None or x.layout == ttnn.ROW_MAJOR_LAYOUT:
        return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
    end = [int(d) - 1 for d in x.shape]
    return ttnn.untilize_with_unpadding(x, output_tensor_end=end, memory_config=memory_config, sub_core_grids=sc)


def sub_cores_tiled(x: ttnn.Tensor):
    """sub_cores(x) for an op that takes sub_core_grids only on TILE inputs (ttnn.typecast: typecast_device_op.cpp
    refuses it for a ROW_MAJOR input); None for a ROW_MAJOR x."""
    return sub_cores(x) if x.layout == ttnn.TILE_LAYOUT else None


def maximum(a: ttnn.Tensor, b: ttnn.Tensor, memory_config=None) -> ttnn.Tensor:
    """ttnn.maximum(a, b) under the cap (FIX.15): the stock binding takes no sub_core_grids (binary_nanobind.cpp
    bind_binary_unary_max_operation), so a capped mesh selects with ttnn.where(ttnn.ge(a, b), a, b) on the capped
    grid: the same values bit for bit (a select, no arithmetic). Without a cap: ttnn.maximum, the call as before."""
    sc = sub_cores(a)
    if sc is None:
        return ttnn.maximum(a, b, memory_config=memory_config)
    ge = ttnn.ge(a, b, memory_config=memory_config, sub_core_grids=sc)
    out = ttnn.where(ge, a, b, memory_config=memory_config, sub_core_grids=sc)
    ttnn.deallocate(ge)
    return out


def from_torch(t, *args, device=None, memory_config=None, **kwargs) -> ttnn.Tensor:
    """ttnn.from_torch under the cap (FIX.15): with a device, ttnn builds the tensor there and converts it with device
    ops on the whole grid (py_to_tt_tensor.cpp can_construct_on_device: tilize, typecast, untilize; 120 cores on the
    Galaxy). A capped mesh builds it on the host (the same conversion, host code) and copies it to the device: no device
    program. Without a cap (or no device): ttnn.from_torch, the call as before."""
    if device is None or max_grid(device) is None:
        return ttnn.from_torch(t, *args, device=device, memory_config=memory_config, **kwargs)
    host = ttnn.from_torch(t, *args, **kwargs)
    return ttnn.to_device(host, device, memory_config=memory_config)


def typecast(x: ttnn.Tensor, dtype, memory_config=None) -> ttnn.Tensor:
    """ttnn.typecast(x, dtype) under the cap (FIX.15): a TILE x on the capped grid; a ROW_MAJOR x, which the stock op
    refuses to place on a sub grid (typecast_device_op.cpp), as tilize -> typecast -> untilize, each on the capped
    grid (the same conversion on the same values; the indexer's [Sq, blocks] fp32 block maxes). Without a cap: the
    call as before."""
    sc = sub_cores(x)
    if sc is None or x.layout == ttnn.TILE_LAYOUT:
        return ttnn.typecast(x, dtype, memory_config=memory_config, sub_core_grids=sc)
    xt = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=memory_config, sub_core_grids=sc)
    yt = ttnn.typecast(xt, dtype, memory_config=memory_config, sub_core_grids=sc)
    ttnn.deallocate(xt)
    y = to_rm(yt, memory_config=memory_config)
    ttnn.deallocate(yt)
    return y
