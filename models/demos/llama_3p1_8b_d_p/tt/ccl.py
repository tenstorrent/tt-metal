# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B CCL manager: every piece of persistent collective state, allocated once.

Owns the CCL sub-device, the reduce-scatter / all-gather ping-pong semaphores, the barrier
semaphores, the three ring-attention semaphores, and the reusable ring-gather scratch buffers.
Attention pulls semaphores and buffers from here per call, so nothing collective is allocated per
layer or per chunk — a 32-layer chunked prefill issues thousands of collectives and allocating
semaphores inside the loop is how this family has previously run the device out of L1.

Each model in this fleet carries its own copy (``minimax_m3/tt/ccl.py`` -> ``gpt_oss_d_p/tt/ccl.py``
-> this). The class body is genuinely model-agnostic, so importing gpt-oss's would work, but it
would put ``models.demos.gpt_oss_d_p`` on the import path of every Llama module that touches a
collective — including the adapter, whose import-light contract ``test_scaffold.py`` asserts. A
140-line copy is the cheaper of the two.
"""

import torch

import ttnn


class CCLManager:
    """Persistent CCL state for one mesh device."""

    def __init__(self, mesh_device, num_links: int = 1, topology: ttnn.Topology = ttnn.Topology.Linear):
        """
        Args:
            topology: ``Linear`` by default, unlike the gpt-oss/M3 copies which default to ``Ring``.
                A ring topology is only correct on a torus-wired pod; asserting it by default on a
                mesh that is not wired that way does not fail cleanly, it deadlocks the collective.
                Callers on a torus pod pass ``Ring`` explicitly.
        """
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology

        # Ring-gather scratch for the ring SDPA, allocated once and reused across every layer and
        # chunk (key -> tensor). See get_ring_gather_buffer.
        self._ring_gather_buffers = {}

        self._init_subdevice()
        self._init_semaphores()
        self.rs_ping_pong_idx = 0
        self.ag_ping_pong_idx = 0
        self.barrier_idx = 0

    def _init_subdevice(self):
        # The REAL device compute grid — Blackhole is wider than 8x8. The ring-attention CCL offset
        # and the ring SDPA program grid must both derive from this same grid, or the op's
        # `ccl_core_grid_offset.x >= sdpa_grid.x` assert fires.
        compute_grid_size = self.mesh_device.compute_with_storage_grid_size()
        self.compute_grid_size = compute_grid_size
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)

        # Ring-attention CCL workers live in the LAST compute column; ring SDPA compute takes the
        # remaining columns. The op requires the two sets to be non-overlapping.
        self.ring_attention_ccl_core_grid_offset = (compute_grid_size.x - 1, 0)

    def _init_semaphores(self):
        # Reduce-scatter: 3 semaphores per cycle, doubled for ping-pong.
        self.rs_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(3 * 2)
        ]
        # All-gather: 2 per cycle, doubled for ping-pong.
        self.ag_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(2 * 2)
        ]
        self.barrier_semaphore = [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(2)]
        # Ring attention: forward/backward all-gather plus a dedicated third counter. Llama never
        # passes a sliding window, so the third is unused by the halo path, but the op's signature
        # takes three and short-changing it reads out of bounds.
        self.ring_attention_ccl_semaphore_handles = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(3)
        ]

    def get_rs_ping_pong_semaphore(self):
        """Reduce-scatter semaphores (3 per cycle), alternating between the two banks."""
        cur_idx = self.rs_ping_pong_idx
        n_sems = 3
        self.rs_ping_pong_idx = (cur_idx + 1) % 2
        return self.rs_ping_pong_semaphores[cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_ag_ping_pong_semaphore(self):
        """All-gather semaphores (2 per cycle), alternating between the two banks."""
        cur_idx = self.ag_ping_pong_idx
        n_sems = 2
        self.ag_ping_pong_idx = (cur_idx + 1) % 2
        return self.ag_ping_pong_semaphores[cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_barrier_semaphore(self):
        cur_idx = self.barrier_idx
        self.barrier_idx = (cur_idx + 1) % 2
        return self.barrier_semaphore[cur_idx]

    def get_ring_gather_buffer(self, key: str, n_kv: int, seq: int, head_dim: int, dtype: ttnn.DataType):
        """Persistent ring-gather scratch for the ring SDPA, allocated once per (key, shape, dtype).

        The op treats it as pure scratch — it fills the gathered region and masks the invalid tail —
        so reuse without re-zeroing is safe. ``key`` separates buffers that are live *at the same
        time* (``"k"`` and ``"v"`` within one op call); shape and dtype key the rest.

        Heads shard across the TP columns and the sequence is replicated across the SP rows
        (``dims=[None, 1]``), which is the layout the ring op reconstructs into.
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

        Deliberately does not touch the barrier or ring-attention semaphores: those are consumed by
        ops that complete within a single call, so they always land back at 0 on their own, and
        zeroing them from the host between chunks would race an in-flight collective.
        """
        for sem in self.rs_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
        for sem in self.ag_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
