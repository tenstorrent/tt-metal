# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.config import Mode


def default_num_links():
    """Default TP-collective link count for the current arch.

    Blackhole boards expose 2 ethernet links between adjacent mesh devices, so
    reduce-scatter / all-gather can run at ~2x bandwidth vs a single link — and
    on Gemma4 prefill the per-layer all-reduces are ~31% of device time, so this
    is the single highest-ROI CCL knob. Wormhole (T3K) defaults to 1 link here
    (its multi-link tuning needs a separate sweep).
    """
    return 2 if is_blackhole() else 1


class CCLManager:
    """CCL manager for Gemma4 tensor parallelism.

    Stores mesh_device reference and num_links for CCL operations.
    Semaphores are retained for the experimental async CCL path (see TODO below).
    """

    def __init__(self, mesh_device, num_links=None, topology=ttnn.Topology.Linear):
        if num_links is None:
            num_links = default_num_links()
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
        self.num_devices = mesh_device.get_num_devices()

        # Semaphores for experimental async CCL ops.
        # TODO: Sweep experimental reduce_scatter_minimal_async + all_gather_async
        # for optimal performance and re-enable. For now we use the simple
        # ttnn.all_reduce / ttnn.all_gather which are functionally correct.
        grid = mesh_device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y
        core_range_set = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)

        self._rs_semaphores = []
        self._ag_semaphores = []
        self._barrier_semaphores = []
        for _ in range(2):
            self._rs_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(3)])
            self._ag_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(2)])
            self._barrier_semaphores.append(ttnn.create_global_semaphore(mesh_device, core_range_set, 0))
        ttnn.synchronize_device(mesh_device)

        self._rs_idx = 0
        self._ag_idx = 0
        self._barrier_idx = 0

        # CP prefill masks, shared by every layer on this mesh and keyed by
        # (local_seq_len, sliding_window). This lives here rather than on each
        # attention module because the mask depends only on the sequence geometry
        # and the layer's window, so a 60-layer stack needs exactly two entries
        # (sliding, global) — not 60 copies of an 8 MiB tensor.
        self._cp_mask_cache = {}

        # ── Ring attention (cross-chunk prefill under CP) ─────────────────────
        # ring_joint SDPA reads the CP-sharded KV cache and gathers the prefix
        # across the CP axis internally with online softmax, so a rank can attend
        # history it does not hold without an explicit AllGather. That is what makes
        # a sharded cache workable for chunk > 0.
        self.compute_grid_size = mesh_device.compute_with_storage_grid_size()
        # CCL workers take the LAST compute column; ring_joint's SDPA compute uses
        # the remaining columns. The op requires the two sets to be disjoint and
        # asserts ccl_core_grid_offset.x < sdpa_grid.x, so both must derive from this
        # same grid (Blackhole is wider than 8x8).
        self.ring_attention_ccl_core_grid_offset = (self.compute_grid_size.x - 1, 0)
        # Forward/backward pair, matching deepseek_v3_d_p and minimax_m3.
        self.ring_attention_ccl_semaphore_handles = [
            ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(2)
        ]
        self._ring_gather_buffers = {}
        # Trace-safe per-chunk scalars for the ring path. One pair for the whole model:
        # slot and prefix length are properties of the chunk, not the layer, so all 60
        # layers read the same two tensors and the host updates them once per chunk.
        self._ring_metadata = None
        # Set by a traced caller to the full context length. logical_n sizes the ring
        # gather at create time and is re-patched per dispatch; a trace does neither, so
        # a per-chunk value would freeze the gather at the capturing chunk's prefix.
        # GEMMA4_RING_LOGICAL_N forces the host logical_n to a fixed value. Used as a
        # probe: on the metadata path the readers derive logical_nt on-device from
        # kv_actual_isl, so a deliberately wrong host value must NOT change results.
        _forced = os.environ.get("GEMMA4_RING_LOGICAL_N")
        self.ring_logical_n_override = int(_forced) if _forced else None

    def _scalar_metadata_tensor(self, value):
        """1-element uint32 replicated DRAM tensor holding one per-chunk scalar.

        Shape/layout/dtype mirror what update_padded_kv_cache and the ring readers
        expect ([1,1,1,1] uint32 row-major in DRAM, replicated so every device reads
        element [0]).
        """
        return ttnn.from_torch(
            torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def get_ring_metadata(self):
        """``(slot_id, kv_actual_isl)`` tensors for the trace-safe ring path.

        Passing these instead of Python ints moves the per-chunk scalars off the host
        dispatch path: the readers load them from DRAM on-device, so the values are not
        baked into the program's runtime args and one captured trace replays across
        chunks. With the scalar form a trace would freeze whichever chunk was live at
        capture, and every later chunk would read the wrong prefix length.
        """
        if self._ring_metadata is None:
            self._ring_metadata = (self._scalar_metadata_tensor(0), self._scalar_metadata_tensor(0))
        return self._ring_metadata

    def set_ring_metadata(self, slot_idx, kv_actual_global):
        """Update the metadata tensors in place for the chunk about to run.

        Called once per chunk, before the layer loop (or before a trace replay). Writes
        into the existing device tensors rather than allocating, because a trace holds
        the addresses it captured.
        """
        slot_t, kv_t = self.get_ring_metadata()
        for tensor, value in ((slot_t, slot_idx), (kv_t, kv_actual_global)):
            host = ttnn.from_torch(
                torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            ttnn.copy_host_to_device_tensor(host, tensor)

    def get_ring_gather_buffer(self, key, n_kv_local, seq, head_dim, dtype):
        """Persistent ring-gather scratch for ``ring_joint`` SDPA.

        Allocated once and reused across every layer and chunk. The op treats it as
        scratch: it fills the gathered region and masks the invalid tail via
        ``kv_actual_isl``, so reuse without re-zeroing is safe.

        ``seq`` must be the FULL cache capacity (max_seq_len), not the current
        ``logical_n``. ring_joint gathers the entire per-device cache shard
        (seq_local = max_seq_len/cp, times cp around the ring), independent of how
        much of it is valid. Sizing to logical_n happens to work when the final
        chunk's logical_n == max_seq_len — i.e. a 2-chunk run — and fails beyond
        that with "gather dim 2 too small" (minimax_m3 hit this at 11 chunks).

        ``key`` separates buffers live in the same call ("k" vs "v"); shape and dtype
        key the rest. Heads shard on the TP columns, sequence replicated across the
        CP rows — the layout the ring op reconstructs into.

        ``n_kv_local`` is the per-device head count. The buffer is built at the global
        size ``n_kv_local * tp_cols`` and sharded across the TP columns so each device
        ends up with its own ``n_kv_local`` heads. Passing the local count straight to
        the sharder fails ("number of chunks N to match the mesh dimension size"), and
        it also has to work for kv-replicated layers where the model's global KV head
        count is smaller than the TP width.
        """
        rows, cols = tuple(self.mesh_device.shape)
        n_kv_global = n_kv_local * cols
        cache_key = (key, n_kv_global, seq, head_dim, str(dtype))
        if cache_key not in self._ring_gather_buffers:
            self._ring_gather_buffers[cache_key] = ttnn.from_torch(
                torch.zeros(1, n_kv_global, seq, head_dim),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=(rows, cols), dims=[None, 1]),
            )
        return self._ring_gather_buffers[cache_key]

    def get_rs_semaphore(self):
        """Returns list of 3 semaphores for reduce_scatter (cycles double-buffer)."""
        sems = self._rs_semaphores[self._rs_idx]
        self._rs_idx = (self._rs_idx + 1) % 2
        return sems

    def get_ag_semaphore(self):
        """Returns list of 2 semaphores for all_gather (cycles double-buffer)."""
        sems = self._ag_semaphores[self._ag_idx]
        self._ag_idx = (self._ag_idx + 1) % 2
        return sems

    def get_barrier_semaphore(self):
        """Returns single barrier semaphore (cycles double-buffer)."""
        sem = self._barrier_semaphores[self._barrier_idx]
        self._barrier_idx = (self._barrier_idx + 1) % 2
        return sem


def cp_degree(mesh_config, mode=Mode.PREFILL):
    """Context-parallel degree along ``mesh_config.sp_axis``; 1 when CP is off.

    CP splits the token dimension across a second mesh axis, so every rank holds
    ``seq_len / cp`` tokens. The degree lives in the ``sp`` field of the mode
    config (named for gpt_oss's sequence parallelism, which is the same idea
    applied to one block).
    """
    if mesh_config is None:
        return 1
    return max(1, mesh_config.get_config(mode).sp)


def ccl_cp_allgather(tensor, mesh_config, ccl_manager, dim, memory_config=None):
    """All-gather along the context-parallel axis.

    Used to rebuild the whole chunk's K/V from the per-rank sequence shards, so
    every rank can attend over all keys while owning only its slice of queries.

    The input must be TILE layout. Tile pages are always 64 B aligned, so this
    takes ttnn's native all_gather; a row-major input whose page is unaligned
    would silently fall back to composite_all_gather, which deadlocks at high
    device counts (see docs/superpowers/specs/2026-08-03-gemma4-context-parallel-prefill-design.md).
    """
    if cp_degree(mesh_config) <= 1:
        return tensor
    assert (
        tensor.layout == ttnn.TILE_LAYOUT
    ), f"ccl_cp_allgather requires TILE layout to stay on the native all_gather path, got {tensor.layout}"
    gathered = ttnn.all_gather(
        tensor,
        dim=dim,
        cluster_axis=mesh_config.sp_axis,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )
    tensor.deallocate(True)
    return gathered


def ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config=None):
    """All-reduce across TP devices."""
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis

    result = ttnn.all_reduce(
        tensor,
        cluster_axis=tp_axis,
        num_links=ccl_manager.num_links,
        topology=ttnn.Topology.Linear,
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    return result

    # TODO: Sweep experimental async reduce_scatter + all_gather for optimal performance.
    # The decomposed path may be faster on T3K but needs tuning of num_links,
    # topology, and num_workers_per_link parameters.
    #
    # scattered = ttnn.experimental.reduce_scatter_minimal_async(
    #     tensor,
    #     dim=3,
    #     cluster_axis=tp_axis,
    #     num_links=ccl_manager.num_links,
    #     topology=ccl_manager.topology,
    #     multi_device_global_semaphore=ccl_manager.get_rs_semaphore(),
    #     barrier_semaphore=ccl_manager.get_barrier_semaphore(),
    #     memory_config=memory_config,
    # )
    # tensor.deallocate(True)
    # gathered = ttnn.experimental.all_gather_async(
    #     scattered,
    #     dim=3,
    #     cluster_axis=tp_axis,
    #     mesh_device=ccl_manager.mesh_device,
    #     num_links=ccl_manager.num_links,
    #     topology=ccl_manager.topology,
    #     multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
    #     barrier_semaphore=ccl_manager.get_barrier_semaphore(),
    #     memory_config=memory_config,
    # )
    # scattered.deallocate(True)
    # return gathered


def ccl_allgather(tensor, mesh_config, ccl_manager, dim=3, memory_config=None):
    """All-gather across TP devices."""
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis

    gathered = ttnn.all_gather(
        tensor,
        dim=dim,
        cluster_axis=tp_axis,
        num_links=ccl_manager.num_links,
        topology=ttnn.Topology.Linear,
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    return gathered

    # TODO: Sweep experimental async all_gather for optimal performance.
    #
    # gathered = ttnn.experimental.all_gather_async(
    #     tensor,
    #     dim=dim,
    #     cluster_axis=tp_axis,
    #     mesh_device=ccl_manager.mesh_device,
    #     num_links=ccl_manager.num_links,
    #     topology=ccl_manager.topology,
    #     multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
    #     barrier_semaphore=ccl_manager.get_barrier_semaphore(),
    #     memory_config=memory_config,
    # )
    # tensor.deallocate(True)
    # return gathered
