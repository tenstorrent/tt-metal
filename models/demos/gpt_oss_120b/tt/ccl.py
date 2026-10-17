# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Persistent CCL state for the [8, 4] mesh (after gpt_oss_d_p/tt/ccl.py): the ring-attention semaphores, the ring
SDPA gather buffers and the core-grid split the ring op needs. Built once per mesh; modules share it."""

import torch

import ttnn

from .settings import settings

SP_AXIS, TP_AXIS = 0, 1


class TtCcl:
    def __init__(self, mesh):
        self.mesh = mesh
        self.num_links = settings.get("CCL_NUM_LINKS")
        self.topology = ttnn.Topology.Ring  # spec box.device_params: FABRIC_1D_RING
        grid = mesh.compute_with_storage_grid_size()
        self.grid = grid
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        # ring SDPA: CCL workers in the last compute column, SDPA on the others (non-overlapping, op requirement)
        self.ring_ccl_offset = ttnn.CoreCoord(grid.x - 1, 0)
        self.ring_sdpa_grid = ttnn.CoreCoord(grid.x - 1, grid.y)
        # forward / backward all-gather plus the sliding-window halo semaphore
        self.ring_semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]
        self._buffers = {}

    def ring_gather_buffer(self, key, n_kv_global, seq, head_dim, dtype):
        """Ring SDPA gather scratch [1, n_kv_global, seq, head_dim]: heads sharded over TP, replicated over SP.
        Allocated once per key; the op fills it and masks the tail, so it is reused without zeroing."""
        k = (key, n_kv_global, seq, head_dim, str(dtype))
        if k not in self._buffers:
            self._buffers[k] = ttnn.from_torch(
                torch.zeros(1, n_kv_global, seq, head_dim),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(None, 1)),
            )
        return self._buffers[k]
