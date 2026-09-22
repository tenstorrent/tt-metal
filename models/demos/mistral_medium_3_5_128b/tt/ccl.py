# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 CCL manager. Structure ported from ``gpt_oss_d_p/tt/ccl.py``.

Owns the CCL sub-device, the reduce-scatter / all-gather ping-pong semaphores, the barrier
semaphores, the three ring-attention semaphores, and the reusable ring-gather scratch buffers. The
attention module takes a ``CCLManager`` and pulls semaphores / buffers from it per call, so all
persistent CCL state is allocated ONCE here (not per layer / per chunk).

This is structural code in the §2.3 sense: the semaphore counts and the ring-attention core-grid
offset are read by the ring SDPA op, so they are copied rather than re-derived. What *is*
re-derived for this model is the ring-gather buffer shape, which carries 2 KV heads per chip here
(8 KV heads / TP=4) where the source carried 1.

Topology default is ``Topology.Linear``: per the recipe, the plain mesh descriptor + FABRIC_1D +
Linear maps on any galaxy, torus-wired or not. The torus is a perf lever behind an env knob
(see ``conftest.py``), never a correctness gate.
"""

import torch

import ttnn


class CCLManager:
    def __init__(self, mesh_device, num_links, topology=ttnn.Topology.Linear):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology

        # Persistent ring-gather scratch buffers for ring SDPA, allocated once and reused across
        # every layer/chunk (key -> tensor). See get_ring_gather_buffer.
        self._ring_gather_buffers = {}

        self._init_subdevice()
        self._init_semaphores()
        self.rs_ping_pong_idx = 0
        self.ag_ping_pong_idx = 0
        self.barrier_idx = 0

    def _init_subdevice(self):
        # Use the REAL device compute grid (Blackhole is wider than 8x8). The ring-attention CCL
        # offset and the ring SDPA program grid must both derive from this same grid, else the op's
        # ccl_core_grid_offset.x >= sdpa_grid.x assert fails.
        compute_grid_size = self.mesh_device.compute_with_storage_grid_size()
        self.compute_grid_size = compute_grid_size
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )

        _worker_sub_device = ttnn.SubDevice(
            [
                self.ccl_cores,
            ]
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)

        # Ring-attention CCL workers live in the LAST compute column; ring SDPA compute uses the
        # remaining columns (the op requires CCL and SDPA cores to be non-overlapping).
        self.ring_attention_ccl_core_grid_offset = (compute_grid_size.x - 1, 0)

    def _init_semaphores(self):
        # Reduce-scatter ping pong: 3 semaphores * 2 buffers
        rs_n_sems = 3 * 2
        self.rs_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(rs_n_sems)
        ]

        # All-gather ping pong: 2 semaphores * 2 buffers
        ag_n_sems = 2 * 2
        self.ag_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(ag_n_sems)
        ]

        barrier_ns_sems = 2 * 1
        self.barrier_semaphore = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(barrier_ns_sems)
        ]

        # Ring-attention semaphores: forward/backward all-gather plus a dedicated third counter.
        self.ring_attention_ccl_semaphore_handles = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(3)
        ]

    def get_rs_ping_pong_semaphore(self):
        """Get semaphores for reduce scatter ping pong operations (3 per cycle)."""
        cur_idx = self.rs_ping_pong_idx
        n_sems = 3
        self.rs_ping_pong_idx = (cur_idx + 1) % 2
        return self.rs_ping_pong_semaphores[cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_ag_ping_pong_semaphore(self):
        """Get semaphores for all gather ping pong operations (2 per cycle)."""
        cur_idx = self.ag_ping_pong_idx
        n_sems = 2
        self.ag_ping_pong_idx = (cur_idx + 1) % 2
        return self.ag_ping_pong_semaphores[cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_barrier_semaphore(self):
        """Get semaphores for barrier operations."""
        cur_idx = self.barrier_idx
        self.barrier_idx = (cur_idx + 1) % 2
        return self.barrier_semaphore[cur_idx]

    def get_ring_gather_buffer(self, key, n_kv, seq, head_dim, dtype):
        """Persistent ring-gather scratch for the ring SDPA — allocated ONCE and reused across every
        layer/chunk. The op treats it as pure scratch: it fills the gathered region and masks the
        invalid tail, so reuse without re-zeroing is safe. ``key`` separates buffers that are live
        simultaneously (e.g. ``"k"`` vs ``"v"`` in one op call); shape/dtype key the rest.

        Heads shard on the TP cols, seq replicated across the SP rows (dims=[None, 1]). For this
        model ``n_kv`` is 8 and TP is 4, so each chip's slice carries **2** KV heads.
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

    def reset_global_semaphores(self):
        """Reset the reduce-scatter / all-gather ping-pong semaphores to 0.

        Unlike the GPT-OSS source this ALSO resets the barrier and ring-attention semaphores: this
        package reuses one CCLManager across the chunks of a chunked prefill (P2), so those counters
        would otherwise carry stale state from chunk to chunk.
        """
        for sem in self.rs_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
        for sem in self.ag_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
        for sem in self.barrier_semaphore:
            ttnn.reset_global_semaphore_value(sem, 0)
        for sem in self.ring_attention_ccl_semaphore_handles:
            ttnn.reset_global_semaphore_value(sem, 0)
