# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Collective-communication manager: sub-device, semaphores and persistent ring buffers.

Structural, copied from ``minimax_m3/tt/ccl.py`` (same mesh, same ops) and verified against the
layout invariants the collectives and the ring SDPA both depend on:

* the CCL core grid is the **real** ``compute_with_storage_grid_size`` — Blackhole is wider than
  8x8, and the ring-SDPA op asserts ``ccl_core_grid_offset.x < sdpa_grid.x``, so a hardcoded 8x8
  here and a real grid there is an op-level assert, not a PCC drop;
* the ring-attention CCL workers live in the LAST compute column and the SDPA program grid takes
  the rest — the op requires them non-overlapping;
* semaphores are handed out in a ping-pong pair so two in-flight collectives never share one.
"""

from __future__ import annotations

import torch

import ttnn


class CCLManager:
    def __init__(self, mesh_device, num_links: int, topology: ttnn.Topology = ttnn.Topology.Linear) -> None:
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology

        self._ring_gather_buffers: dict[tuple, ttnn.Tensor] = {}
        self._init_subdevice()
        self._init_semaphores()
        self.rs_ping_pong_idx = 0
        self.ag_ping_pong_idx = 0
        self.barrier_idx = 0

    def _init_subdevice(self) -> None:
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.compute_grid_size = grid
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)
        # Ring-attention CCL workers in the last compute column; ring-joint SDPA compute uses the rest.
        self.ring_attention_ccl_core_grid_offset = (grid.x - 1, 0)

    def _init_semaphores(self) -> None:
        self.rs_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(3 * 2)
        ]
        self.ag_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(2 * 2)
        ]
        self.barrier_semaphore = [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(2)]
        self.ring_attention_ccl_semaphore_handles = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(2)
        ]

    def get_rs_ping_pong_semaphore(self) -> list:
        cur = self.rs_ping_pong_idx
        self.rs_ping_pong_idx = (cur + 1) % 2
        return self.rs_ping_pong_semaphores[cur * 3 : (cur + 1) * 3]

    def get_ag_ping_pong_semaphore(self) -> list:
        cur = self.ag_ping_pong_idx
        self.ag_ping_pong_idx = (cur + 1) % 2
        return self.ag_ping_pong_semaphores[cur * 2 : (cur + 1) * 2]

    def get_barrier_semaphore(self):
        cur = self.barrier_idx
        self.barrier_idx = (cur + 1) % 2
        return self.barrier_semaphore[cur]

    def get_ring_gather_buffer(self, key: str, n_kv: int, seq: int, head_dim: int, dtype) -> ttnn.Tensor:
        """Persistent ring-gather scratch for ``ring_joint`` SDPA, allocated once and reused across
        every layer and chunk. The op treats it as pure scratch — it fills the gathered region and
        masks the invalid tail via ``kv_actual_isl`` — so reuse without re-zeroing is safe. Heads
        shard on the TP cols, sequence replicated across the SP rows: the layout the op rebuilds into.
        """
        cache_key = (key, n_kv, seq, head_dim, str(dtype))
        if cache_key not in self._ring_gather_buffers:
            rows, cols = tuple(self.mesh_device.shape)
            self._ring_gather_buffers[cache_key] = ttnn.from_torch(
                torch.zeros(1, n_kv, seq, head_dim),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=(rows, cols), dims=[None, 1]),
            )
        return self._ring_gather_buffers[cache_key]

    def reset_global_semaphores(self) -> None:
        for sem in self.rs_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
        for sem in self.ag_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
