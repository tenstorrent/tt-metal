# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Ring-attention CCL state for CP=4 on the 1x4 mesh (FABRIC_2D, Linear topology on the CP axis).

Adapted from models/demos/gpt_oss_d_p/tt/ccl.py (CCLManager: ring semaphores, persistent ring-gather buffers, CCL
column). Everything is allocated once and reused across layers and chunks.
"""

from __future__ import annotations

import ttnn

CP_AXIS = 1  # mesh [1, 4]: the sequence is split over the columns


class RingCCL:
    def __init__(self, mesh, num_links: int = 1, topology=ttnn.Topology.Linear):
        self.mesh, self.num_links, self.topology = mesh, num_links, topology
        self.cp_axis = CP_AXIS
        self.cp = tuple(mesh.shape)[CP_AXIS]
        grid = mesh.compute_with_storage_grid_size()
        self.compute_grid_size = grid
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        # Ring CCL workers take the last compute column; the ring SDPA grid is the rest (they must not overlap).
        self.ccl_core_grid_offset = ttnn.CoreCoord(grid.x - 1, 0)
        self.sdpa_grid = ttnn.CoreCoord(grid.x - 1, grid.y)
        # forward / backward all-gather + the halo semaphore (the op takes three).
        self.ring_semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]
        self._buffers = {}

    def gather_buffer(self, key: str, n_kv: int, seq: int, head_dim: int, dtype) -> ttnn.Tensor:
        """Persistent [1, n_kv, seq, head_dim] ring-gather scratch, replicated (TP=1: every chip holds every KV
        head). Zeroed once when first asked for (load time: the KV caches ask when they are built); the op fills the
        gathered region itself, so it is never re-zeroed."""
        k = (key, n_kv, seq, head_dim, str(dtype))
        if k not in self._buffers:
            self._buffers[k] = ttnn.zeros(
                [1, n_kv, seq, head_dim],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._buffers[k]

    def free(self):
        for t in self._buffers.values():
            ttnn.deallocate(t)
        self._buffers = {}
