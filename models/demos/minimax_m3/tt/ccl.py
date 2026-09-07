# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


class CCLManager:
    """Semaphores, topology and persistent scratch for M3's collectives.

    ``topology`` is handed to the LEGACY CCLs (all_gather_async, reduce_scatter_minimal_async, ring_joint).
    The galaxy prefill runtime passes Linear, also on the FABRIC_1D_RING fabric it now runs on: Ring on the
    legacy CCLs has faster kernels but more dispatch overhead and was a net loss untraced (row 4 in
    docs/ATTENTION_HIGH_BW_ALL_GATHER.md). Revisit together with FABRIC_2D_TORUS_XY once prefill runs under
    trace and the 2D-optimised MoE dispatch/combine land. ``high_bw_all_gather`` ignores this topology: it
    derives ring vs line from the fabric itself.
    """

    def __init__(self, mesh_device, num_links, topology=ttnn.Topology.Ring):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology

        # Cache for ping pong buffers: key = (shape_tuple, dim, mesh_axis), value = [buffer1, buffer2]
        self._ping_pong_buffer_cache = {}
        self._ping_pong_buffer_indices = {}

        # Persistent ring-gather scratch buffers for ring_joint SDPA, allocated once and reused across
        # every layer/chunk (key -> tensor). See get_ring_gather_buffer.
        self._ring_gather_buffers = {}

        # Persistent output buffers for ttnn.experimental.high_bw_all_gather (the MSA K/V/index_k SP
        # gathers). The op writes a caller-owned, fixed worst-case-shape DRAM output; layers run serially,
        # so one buffer per (key, shape, dtype) is shared across every layer/chunk. See
        # get_high_bw_gather_buffer.
        self._high_bw_gather_buffers = {}

        # Setup semaphores
        self._init_subdevice()

        # Initialize semaphores for reduce scatter and all gather
        self._init_semaphores()
        self.rs_ping_pong_idx = 0
        self.ag_ping_pong_idx = 0
        self.barrier_idx = 0

    def _init_subdevice(self):
        # Use the REAL device compute grid (Blackhole is wider than 8x8). The ring-attention CCL offset
        # and the ring_joint SDPA program grid must both derive from this same grid, else the op's
        # ccl_core_grid_offset.x >= sdpa_grid.x assert fails. Matches deepseek_v3_d_p tt_ccl.
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

        # Ring-attention CCL workers live in the LAST compute column; ring_joint SDPA compute uses
        # the remaining columns (the op requires CCL and SDPA cores to be non-overlapping). Mirrors
        # deepseek_v3_d_p tt_ccl.ring_attention_ccl_core_grid_offset = (grid.x - 1, 0). Used by the
        # SP=8 dense-attention path (ring_joint_scaled_dot_product_attention); see PREFILL_PROPOSAL §6.6.
        self.ring_attention_ccl_core_grid_offset = (compute_grid_size.x - 1, 0)

    def _init_semaphores(self):
        # Initialize semaphores for reduce scatter ping pong
        rs_n_sems = 3 * 2  # 3 semaphores * 2 for ping pong
        self.rs_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(rs_n_sems)
        ]

        # Initialize semaphores for all gather ping pong
        ag_n_sems = 2 * 2  # 2 semaphores * 2 for ping pong (2 buffers)
        self.ag_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(ag_n_sems)
        ]

        # Initialize barrier semaphores
        barrier_ns_sems = 2 * 1
        self.barrier_semaphore = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(barrier_ns_sems)
        ]

        # Ring-attention semaphores: a forward/backward PAIR for ring_joint_scaled_dot_product_attention
        # (the SP=8 dense-attention path). Matches deepseek_v3_d_p create_global_semaphores (2 handles).
        self.ring_attention_ccl_semaphore_handles = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(2)
        ]

    def get_rs_ping_pong_semaphore(self):
        """
        Get semaphores for reduce scatter ping pong operations.

        Returns:
            List of 3 semaphores for the current ping pong cycle
        """
        cur_idx = self.rs_ping_pong_idx
        n_sems = 3
        self.rs_ping_pong_idx = (cur_idx + 1) % 2
        return self.rs_ping_pong_semaphores[cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_ag_ping_pong_semaphore(self):
        """
        Get semaphores for all gather ping pong operations.

        Returns:
            List of 2 semaphores for the current ping pong cycle
        """
        cur_idx = self.ag_ping_pong_idx
        n_sems = 2
        self.ag_ping_pong_idx = (cur_idx + 1) % 2
        return self.ag_ping_pong_semaphores[cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_barrier_semaphore(self):
        """
        Get semaphores for barrier operations.
        """
        cur_idx = self.barrier_idx
        self.barrier_idx = (cur_idx + 1) % 2
        return self.barrier_semaphore[cur_idx]

    def get_ring_gather_buffer(self, key, n_kv, seq, head_dim, dtype):
        """Persistent ring-gather scratch for ``ring_joint`` SDPA — allocated ONCE and reused across every
        layer/chunk (replaces the per-call ``from_torch(zeros)`` that churned host + DRAM on every dense
        attention). The op treats it as pure scratch: it fills the gathered region and masks the invalid
        tail via ``kv_actual_isl``, so reuse without re-zeroing is safe (matches DeepSeek's shared TT_CCL
        ring buffers). ``key`` separates buffers that are live simultaneously (e.g. ``"k"`` vs ``"v"`` in one
        op call); shape/dtype key the rest (cache-read is bf8 x max_seq_len, first-chunk is bf16 x chunk).
        Heads shard on the TP cols, seq replicated across the SP rows (dims=[None, 1]) — the layout the
        ring op reconstructs into.
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

    def get_high_bw_gather_buffer(self, key, shape, dtype, layout=ttnn.TILE_LAYOUT):
        """Persistent, REPLICATED, DRAM-interleaved output for ``ttnn.experimental.high_bw_all_gather``.

        The op has no allocation of its own: it writes into a caller-provided output whose shape is the
        WORST-CASE gathered shape (input dim x axis size), landing rank r's rows at the fixed slot
        ``r * input_rows`` regardless of how much of the input is active (``gathered_dim_size``). The
        buffer is therefore sized once for the full cache capacity and reused for every chunk/layer; the
        never-written tail is scratch the consumers must not dereference (the MSA indexer is bounded by
        ``kv_len``, sparse SDPA only reads selected blocks). ``key`` separates buffers that are live at the
        same time (K vs V vs index_k in one attention call); ``shape``/``dtype`` key the rest. Zero-filled
        once at allocation so an accidental read of the tail is finite rather than NaN garbage.
        Mirrors DeepSeek's ``TT_CCL.get_mla_sparse_kv_gather_buffer``.
        """
        cache_key = (key, tuple(shape), str(dtype), str(layout))
        if cache_key not in self._high_bw_gather_buffers:
            self._high_bw_gather_buffers[cache_key] = ttnn.from_torch(
                torch.zeros(*shape),
                dtype=dtype,
                layout=layout,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        return self._high_bw_gather_buffers[cache_key]

    def reset_global_semaphores(self):
        """Reset all global semaphores to 0"""
        for sem in self.rs_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
        for sem in self.ag_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
