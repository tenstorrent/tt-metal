# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Collectives of the [8, 4] mesh (axis 0 = SP, axis 1 = TP) and the ring-attention resources shared by layers.

The residual stream holds 160 rows per chip; a column-parallel projection first gathers its SP row's 640 rows over TP
(``gather_rows``) and a row-parallel one sums its TP partials and keeps this chip's 160 rows (``scatter_rows``): an
all-reduce split around the norms and residual adds (gemma4_d_p ``ccl_allgather`` / ``ccl_reduce_scatter_rows``).
"""

from __future__ import annotations

import ttnn

from .settings import settings

SP_AXIS, TP_AXIS = 0, 1


def gather_rows(x, memory_config=None):
    """[1, 1, 160, H] local rows -> [1, 1, 640, H]: the SP row's tokens, in TP column order."""
    return ttnn.all_gather(x, dim=2, cluster_axis=TP_AXIS, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)


def scatter_rows(x, memory_config=None):
    """[1, 1, 640, H] TP partials -> [1, 1, 160, H]: their sum, this chip's rows."""
    return ttnn.reduce_scatter(x, dim=2, cluster_axis=TP_AXIS, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)


class RingResources:
    """Global semaphores, the CCL core column and persistent gather buffers of the ring SDPA over SP (axis 0).

    The CCL workers take the last compute column; the SDPA computes on the others (the op asserts they are
    disjoint). Buffers are keyed so layers of the same shape share one."""

    def __init__(self, mesh):
        self.mesh = mesh
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, row_wise=True)
        self.sdpa_grid = ttnn.CoreCoord(grid.x - 1, grid.y)
        self.ccl_core_grid_offset = ttnn.CoreCoord(grid.x - 1, 0)
        # forward / backward all-gather and the neighbour-halo exchange
        self.semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]
        ttnn.synchronize_device(mesh)  # load time: the semaphores are initialised before any op uses them
        self.num_links = settings.get("RING_NUM_LINKS")
        self._buffers = {}

    def gather_buffer(self, key, heads: int, seq: int, width: int, dtype):
        k = (key, heads, seq, width, str(dtype))
        if k not in self._buffers:
            self._buffers[k] = ttnn.zeros(
                [1, heads, seq, width],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._buffers[k]
