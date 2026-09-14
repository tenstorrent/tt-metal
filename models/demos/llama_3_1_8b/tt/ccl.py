# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CCL manager: global semaphores, the CCL sub-device, and the ring-SDPA scratch buffers.

Structural, copied deliberately from ``minimax_m3/tt/ccl.py`` and verified against the invariants
the ops actually enforce (recipe §2.3):

* the CCL core grid is derived from the device's REAL ``compute_with_storage_grid_size`` — on
  Blackhole that is wider than 8x8, and ``ring_joint_scaled_dot_product_attention`` asserts
  ``ccl_core_grid_offset.x >= sdpa_grid.x``, so both must come from the same grid or the op refuses
  to build;
* ring-attention CCL workers live in the LAST compute column and the SDPA program config gets
  ``grid.x - 1``, because the op requires the two core sets to be disjoint;
* collectives use PING-PONG semaphore pairs, so two in flight on the same axis cannot alias.

The ring-gather scratch is allocated once and reused for every layer and chunk. The op treats it as
pure scratch (it fills the gathered region and masks the invalid tail via ``kv_actual_isl``), so
reuse without re-zeroing is safe — and a ``from_torch(zeros)`` per call would otherwise churn host
and DRAM 32 times per chunk.
"""

from __future__ import annotations

import torch

import ttnn


class CCLManager:
    def __init__(self, mesh_device, num_links: int, topology=ttnn.Topology.Linear):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology

        grid = mesh_device.compute_with_storage_grid_size()
        self.compute_grid_size = grid
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)
        # Ring-attention CCL workers in the last compute column; SDPA compute gets the rest.
        self.ring_attention_ccl_core_grid_offset = (grid.x - 1, 0)

        self.rs_ping_pong_semaphores = [ttnn.create_global_semaphore(mesh_device, self.ccl_cores, 0) for _ in range(6)]
        self.ag_ping_pong_semaphores = [ttnn.create_global_semaphore(mesh_device, self.ccl_cores, 0) for _ in range(4)]
        self.barrier_semaphore = [ttnn.create_global_semaphore(mesh_device, self.ccl_cores, 0) for _ in range(2)]
        # A forward/backward PAIR for ring_joint_scaled_dot_product_attention.
        self.ring_attention_ccl_semaphore_handles = [
            ttnn.create_global_semaphore(mesh_device, self.ccl_cores, 0) for _ in range(2)
        ]

        self.rs_ping_pong_idx = 0
        self.ag_ping_pong_idx = 0
        self.barrier_idx = 0
        self._ring_gather_buffers = {}

    def get_rs_ping_pong_semaphore(self):
        i, n = self.rs_ping_pong_idx, 3
        self.rs_ping_pong_idx = (i + 1) % 2
        return self.rs_ping_pong_semaphores[i * n : (i + 1) * n]

    def get_ag_ping_pong_semaphore(self):
        i, n = self.ag_ping_pong_idx, 2
        self.ag_ping_pong_idx = (i + 1) % 2
        return self.ag_ping_pong_semaphores[i * n : (i + 1) * n]

    def get_barrier_semaphore(self):
        i = self.barrier_idx
        self.barrier_idx = (i + 1) % 2
        return self.barrier_semaphore[i]

    def get_ring_gather_buffer(self, key, n_kv_global, seq, head_dim, dtype):
        """Persistent ring-gather scratch for ``ring_joint`` SDPA, allocated once per distinct shape.

        ``n_kv_global`` is the GLOBAL KV-head count (8 for Llama-3.1-8B); ``dims=[None, 1]`` shards
        it across the TP cols so each chip sees the 2 heads it actually holds, and replicates the
        sequence across the SP rows — the layout the ring op reconstructs into. ``key`` separates
        buffers that are live at the same time ("k" vs "v" inside one op call).
        """
        cache_key = (key, n_kv_global, seq, head_dim, str(dtype))
        if cache_key not in self._ring_gather_buffers:
            rows, cols = tuple(self.mesh_device.shape)
            self._ring_gather_buffers[cache_key] = ttnn.from_torch(
                torch.zeros(1, n_kv_global, seq, head_dim),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=(rows, cols), dims=[None, 1]),
            )
        return self._ring_gather_buffers[cache_key]

    def reset_global_semaphores(self):
        for sem in self.rs_ping_pong_semaphores + self.ag_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
