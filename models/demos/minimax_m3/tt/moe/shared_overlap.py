# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared expert concurrent with the routed-expert dispatch, on disjoint Tensix sub-devices.

The deepseek_v3_d_p recipe (tt/moe/tt_moe.py, overlap_shared_expert_with_dispatch): split the grid into a
dispatch strip (the top row, which holds the workers nearest the eth cores) and a shared-expert strip (the
rest), load that sub-device manager around dispatch + the shared expert's matmuls, clear it after.

  sub-device 0 (dispatch): rows [0, DISPATCH_ROWS)
  sub-device 1 (shared):   rows [DISPATCH_ROWS, grid_y)

Rules inside the window: every op names a sub-device (or sub_core_grids); no collective runs in it (the
shared expert's TP collective runs after the clear, or is fused into the routed reduce-scatter); no buffer an
op of either sub-device still uses is freed in it. The caller collects those in a keep-alive list and frees
them after the clear. Loading and clearing each drain the device.

The manager is created once per mesh and shared by every MoE layer on it. ``release(mesh_device)`` removes it
and must run before the mesh is closed (Model.release_sub_device_managers does it); a later load() re-creates it.

Both schedules are on by default (like deepseek_v3_d_p's overlap). The overlap follows the prefill runner's
``PREFILL_OVERLAP_SHARED_EXPERT`` (TtPrefillRuntimeConfig.overlap_shared_expert); off, the shared expert runs before
the MoE on the full grid. ``M3_MOE_FUSE_SHARED_RS=0`` gives it back its own TP collective.
"""

import functools
import math
import os

import ttnn

DEFAULT_FUSE_SHARED_RS = True

# dispatch puts its senders in its sub-device's first row.
DISPATCH_ROWS = 1

# id(mesh device) -> _Manager: one manager per mesh, shared by every MoE layer on it. The mesh is held so its
# id cannot be reused by another mesh while the entry exists; release() drops the entry (MeshDevice does not
# support weak references).
_MANAGERS = {}

# Circular-buffer budget per core for the sub-device matmuls, in bf16 tiles (2 KiB): 768 KiB of Blackhole's
# 1.5 MiB L1. M3's weights are bf16, so deepseek's get_program_configs (sized for its shapes) overflows L1
# here: the down projection's full-K in1 block alone is 24 x 18 tiles.
CB_BUDGET_TILES = 384
MAX_IN0_BLOCK_W = 16
DEST_TILES = 8  # bf16 dest, half-sync: the default compute config of ttnn.linear for a bf16 output


def _flag(var, default):
    value = os.getenv(var)
    if value is None or not value.strip():
        return default
    value = value.strip().lower()
    if value not in ("0", "1"):
        raise ValueError(f"{var} must be 0 or 1 (got {value!r})")
    return value == "1"


def fuse_shared_rs_enabled():
    """Add the shared expert's un-reduced partial to the routed output before the MoE's reduce-scatter instead
    of running its own TP collective. Mathematically equivalent (the reduce-scatter is linear); not
    bit-identical, the added bf16 add rounds before the collective. ``M3_MOE_FUSE_SHARED_RS=0|1``."""
    return _flag("M3_MOE_FUSE_SHARED_RS", DEFAULT_FUSE_SHARED_RS)


def split_grid(grid_x, grid_y, dispatch_rows):
    """(dispatch cores, shared cores) CoreRangeSets: rows [0, dispatch_rows) and [dispatch_rows, grid_y)."""
    assert 0 < dispatch_rows < grid_y, f"dispatch_rows={dispatch_rows} must be in (0, grid_y={grid_y})"
    dispatch = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, dispatch_rows - 1))})
    shared = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, dispatch_rows), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}
    )
    return dispatch, shared


def _divisors_desc(n, cap=None):
    return [d for d in range(min(n, cap or n), 0, -1) if n % d == 0]


def cb_tiles(in0_block_w, out_block_h, out_block_w, k_tiles):
    """Tiles the 2D mcast factory allocates per core for interleaved bf16 in/out: in0 and in1 blocks
    (double-buffered when K takes more than one block) and one output block (interm0 shares it)."""
    depth = 2 if k_tiles > in0_block_w else 1
    return depth * (out_block_h + out_block_w) * in0_block_w + out_block_h * out_block_w


def _block_cost(k_tiles, per_core_M, per_core_N, obh, obw, w):
    """Rough per-core cost in tile units: the multiplies, the in0 / in1 tiles read (in0 is re-read for every
    output-block column, in1 for every block row) and a fixed overhead per K-block iteration."""
    nbx, nby = per_core_N // obw, per_core_M // obh
    reads = k_tiles * (per_core_M * nbx + per_core_N * nby)
    return per_core_M * per_core_N * k_tiles / 2 + reads + 64 * (k_tiles // w) * nbx * nby


@functools.lru_cache(maxsize=None)
def _matmul_2d_blocks(grid_x, grid_y, m_tiles, k_tiles, n_tiles, budget_tiles):
    """(per_core_M, per_core_N, out_block_h, out_block_w, in0_block_w, out_subblock_h, out_subblock_w): the
    cheapest blocking by _block_cost whose CBs fit budget_tiles. Pure in its args, so cached: every sparse layer
    asks for the same few shapes on every forward."""
    best = None
    m0, n0 = math.ceil(m_tiles / grid_y), math.ceil(n_tiles / grid_x)
    for per_core_M in range(m0, m0 + 3):
        for per_core_N in range(n0, n0 + 2):
            for obh in _divisors_desc(per_core_M):
                for obw in _divisors_desc(per_core_N):
                    for w in _divisors_desc(k_tiles, MAX_IN0_BLOCK_W):
                        if cb_tiles(w, obh, obw, k_tiles) > budget_tiles:
                            continue
                        cost = _block_cost(k_tiles, per_core_M, per_core_N, obh, obw, w)
                        if best is None or cost < best[0]:
                            best = (cost, per_core_M, per_core_N, obh, obw, w)
                        break  # the deepest K block that fits is the cheapest for this output block
    assert best is not None, f"no 2D matmul block fits {budget_tiles} tiles (M={m_tiles} K={k_tiles} N={n_tiles})"
    _, per_core_M, per_core_N, obh, obw, w = best
    sub = max(
        ((h, sw) for h in _divisors_desc(obh) for sw in _divisors_desc(obw) if h * sw <= DEST_TILES),
        key=lambda hw: (hw[0] * hw[1], hw[1]),
    )
    return per_core_M, per_core_N, obh, obw, w, sub[0], sub[1]


def matmul_2d_config(grid, m_tiles, k_tiles, n_tiles, budget_tiles=CB_BUDGET_TILES):
    """2D mcast config over `grid` (M over rows, N over columns) whose CBs fit budget_tiles, the cheapest
    by _block_cost. per_core_M / per_core_N may round up past the minimum when that gives better blocks
    (the launched grid is then smaller than `grid`)."""
    per_core_M, per_core_N, obh, obw, w, sub_h, sub_w = _matmul_2d_blocks(
        grid.x, grid.y, m_tiles, k_tiles, n_tiles, budget_tiles
    )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=obh,
        out_block_w=obw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def shared_expert_program_configs(cores, x, gate_proj, down_proj):
    """(gate, up, down) 2D matmul configs sized to the shared sub-device's grid. The gate keeps its raw
    accumulator: M3's clamped swigluoai is applied by one fused multiply after it."""
    grid = cores.bounding_box().grid_size()
    t = ttnn.TILE_SIZE
    m = x.padded_shape[-2] // t
    gate = matmul_2d_config(grid, m, gate_proj.padded_shape[-2] // t, gate_proj.padded_shape[-1] // t)
    down = matmul_2d_config(grid, m, down_proj.padded_shape[-2] // t, down_proj.padded_shape[-1] // t)
    return gate, gate, down


class _Manager:
    def __init__(self, mesh_device, manager_id):
        self.mesh_device = mesh_device
        self.manager_id = manager_id
        self.loaded = False


def _acquire(mesh_device, dispatch_cores, shared_cores):
    """The mesh's overlap manager, created on first use."""
    entry = _MANAGERS.get(id(mesh_device))
    if entry is None:
        manager_id = mesh_device.create_sub_device_manager(
            [ttnn.SubDevice([dispatch_cores]), ttnn.SubDevice([shared_cores])], 0
        )
        entry = _MANAGERS[id(mesh_device)] = _Manager(mesh_device, manager_id)
    return entry


def release(mesh_device):
    """Remove mesh_device's overlap manager, clearing it first if it is loaded. No-op if there is none. Call
    before closing the mesh: a sub-device manager left registered at close can segfault the teardown."""
    entry = _MANAGERS.get(id(mesh_device))
    if entry is None:
        return
    if entry.loaded:
        mesh_device.clear_loaded_sub_device_manager()
        entry.loaded = False
    mesh_device.remove_sub_device_manager(entry.manager_id)
    del _MANAGERS[id(mesh_device)]


def release_all():
    """release() every mesh that has an overlap manager."""
    for entry in list(_MANAGERS.values()):
        release(entry.mesh_device)


class SharedExpertOverlap:
    """The sub-device split and its load / clear."""

    def __init__(self, mesh_device):
        grid = mesh_device.compute_with_storage_grid_size()
        self.mesh_device = mesh_device
        self.dispatch_cores, self.shared_cores = split_grid(grid.x, grid.y, DISPATCH_ROWS)
        _acquire(mesh_device, self.dispatch_cores, self.shared_cores)
        self.dispatch_sd_id = ttnn.SubDeviceId(0)
        self.shared_sd_id = ttnn.SubDeviceId(1)

    def _entry(self):
        return _acquire(self.mesh_device, self.dispatch_cores, self.shared_cores)

    @property
    def manager_id(self):
        return self._entry().manager_id

    @property
    def shared_sub_device(self):
        """(sub-device id, cores): what the shared expert's ops are confined to."""
        return self.shared_sd_id, self.shared_cores

    def load(self):
        entry = self._entry()
        self.mesh_device.load_sub_device_manager(entry.manager_id)
        entry.loaded = True

    def clear(self):
        self.mesh_device.clear_loaded_sub_device_manager()
        entry = _MANAGERS.get(id(self.mesh_device))
        if entry is not None:
            entry.loaded = False
