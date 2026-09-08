# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole


def default_num_links():
    # GEMMA4_NUM_LINKS overrides the CCL link count. Used to test whether the ring
    # replay deadlock involves the two parallel link workers racing each other.
    import os as _os

    _forced = _os.environ.get("GEMMA4_NUM_LINKS")
    if _forced:
        return int(_forced)
    return _default_num_links_impl()


def _default_num_links_impl():
    """Default TP-collective link count for the current arch.

    Blackhole boards expose 2 ethernet links between adjacent mesh devices, so
    reduce-scatter / all-gather can run at ~2x bandwidth vs a single link — and
    on Gemma4 prefill the per-layer all-reduces are ~31% of device time, so this
    is the single highest-ROI CCL knob. Wormhole (T3K) defaults to 1 link here
    (its multi-link tuning needs a separate sweep).

    Override with ``GEMMA4_CCL_NUM_LINKS``.
    """
    env = os.environ.get("GEMMA4_CCL_NUM_LINKS")
    if env is not None:
        return max(1, int(env))
    return 2 if is_blackhole() else 1


def ccl_chunks_per_sync() -> int:
    """Async RS/AG ``chunks_per_sync`` (fabric packet grouping). Default 10."""
    return max(1, int(os.environ.get("GEMMA4_CCL_CHUNKS_PER_SYNC", "10")))


def ccl_num_workers_per_link() -> int:
    """Async RS/AG workers per link. Default 2."""
    return max(1, int(os.environ.get("GEMMA4_CCL_NUM_WORKERS", "2")))


def ccl_num_buffers_per_channel() -> int:
    """Async RS/AG ``num_buffers_per_channel``. Default 2."""
    return max(1, int(os.environ.get("GEMMA4_CCL_NUM_BUFFERS", "2")))


def ccl_persistent_buffers_enabled() -> bool:
    """Reuse DRAM destination buffers across RS/AG calls (Phase P1).

    Default on for async path; disable with ``GEMMA4_CCL_PERSISTENT_BUF=0``.
    Sync ``ttnn.all_reduce`` ignores this (no persistent buffer API).
    """
    return os.environ.get("GEMMA4_CCL_PERSISTENT_BUF", "1").lower() not in ("0", "false", "no")


def default_ccl_topology():
    """Use ring collectives on Galaxy unless linear topology is requested."""
    override = os.environ.get("GEMMA4_CCL_TOPOLOGY", "").strip().lower()
    return ttnn.Topology.Linear if override in ("linear", "line", "l") else ttnn.Topology.Ring


def ccl_async_enabled() -> bool:
    """True when prefill/decode allreduce should use async RS+AG.

    Default off until measured green on the target board; enable with
    ``GEMMA4_CCL_ASYNC=1``.
    """
    return os.environ.get("GEMMA4_CCL_ASYNC", "0").lower() in ("1", "true", "yes")


class CCLManager:
    """CCL manager for Gemma4 tensor parallelism.

    Stores mesh_device, num_links, and topology for CCL operations.
    Semaphores support the async RS+AG path (``GEMMA4_CCL_ASYNC=1``).
    Persistent DRAM buffers (``GEMMA4_CCL_PERSISTENT_BUF``) are keyed by shape
    so repeated collectives of the same activation shape skip realloc+barrier.
    """

    def __init__(self, mesh_device, num_links=None, topology=None):
        if num_links is None:
            num_links = default_num_links()
        if topology is None:
            topology = default_ccl_topology()
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
        self.num_devices = mesh_device.get_num_devices()
        topo_name = "Ring" if topology == ttnn.Topology.Ring else "Linear"
        logger.info(
            f"Gemma4 CCLManager: devices={self.num_devices} num_links={num_links} "
            f"topology={topo_name} async={int(ccl_async_enabled())} "
            f"persistent_buf={int(ccl_persistent_buffers_enabled())}"
        )

        grid = mesh_device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y
        core_range_set = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)

        self._rs_semaphores = []
        self._ag_semaphores = []
        self._barrier_semaphores = []
        for _ in range(2):
            self._rs_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(3)])
            self._ag_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(3)])
            self._barrier_semaphores.append(ttnn.create_global_semaphore(mesh_device, core_range_set, 0))
        ttnn.synchronize_device(mesh_device)

        self._rs_idx = 0
        self._ag_idx = 0
        self._barrier_idx = 0
        # shape_key -> ttnn.Tensor (DRAM interleaved zeros)
        self._persistent_rs_out: dict = {}
        self._persistent_rs_inter: dict = {}

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
        # THREE, not the usual forward/backward pair. The third is the neighbor-halo
        # exchange's own counter. With only two, the halo reuses semaphores[0] — the
        # all-gather's backward semaphore — and lands on the same worker core, so two
        # protocols with different arrival counts share one counter. The halo's completion
        # then destroys all-gather increments and the ring deadlocks at depth. See
        # docs/superpowers/specs/2026-08-06-ring-trace-replay-deadlock.md.
        self.ring_attention_ccl_semaphore_handles = [
            ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(3)
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

    def get_ring_gather_buffer(self, key, n_kv_local, seq, head_dim, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG):
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
        cache_key = (key, n_kv_global, seq, head_dim, str(dtype), str(memory_config))
        if cache_key not in self._ring_gather_buffers:
            self._ring_gather_buffers[cache_key] = ttnn.from_torch(
                torch.zeros(1, n_kv_global, seq, head_dim),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=memory_config,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=(rows, cols), dims=[None, 1]),
            )
        return self._ring_gather_buffers[cache_key]

    def get_rs_semaphore(self):
        """Returns list of 3 semaphores for reduce_scatter (cycles double-buffer)."""
        sems = self._rs_semaphores[self._rs_idx]
        self._rs_idx = (self._rs_idx + 1) % 2
        return sems

    def get_ag_semaphore(self):
        """Returns list of 3 semaphores for all_gather (cycles double-buffer)."""
        sems = self._ag_semaphores[self._ag_idx]
        self._ag_idx = (self._ag_idx + 1) % 2
        return sems

    def get_barrier_semaphore(self):
        """Returns single barrier semaphore (cycles double-buffer)."""
        sem = self._barrier_semaphores[self._barrier_idx]
        self._barrier_idx = (self._barrier_idx + 1) % 2
        return sem

    def _shape_key(self, shape, dtype, memory_config):
        return (tuple(int(x) for x in shape), str(dtype), str(memory_config))

    def get_persistent_rs_buffers(self, tensor, memory_config, tp):
        if not ccl_persistent_buffers_enabled():
            return None
        if tp <= 1:
            return None
        # Reduce-scatter shrinks dim=3 by TP group size (not full mesh size).
        out_shape = list(tensor.shape)
        out_shape[3] = int(out_shape[3]) // tp
        # Linear topology needs a leading size-2 dim for forward/backward streams.
        inter_shape = list(tensor.shape)
        if self.topology == ttnn.Topology.Linear:
            inter_shape = [2] + inter_shape
        inter_key = self._shape_key(inter_shape, tensor.dtype, ttnn.DRAM_MEMORY_CONFIG)
        out_key = self._shape_key(out_shape, tensor.dtype, memory_config)
        inter = self._persistent_rs_inter.get(inter_key)
        if inter is None:
            inter = ttnn.zeros(
                inter_shape,
                dtype=tensor.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._persistent_rs_inter[inter_key] = inter
        out = self._persistent_rs_out.get(out_key)
        if out is None:
            out = ttnn.zeros(
                out_shape,
                dtype=tensor.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=memory_config,
            )
            self._persistent_rs_out[out_key] = out
            logger.debug(f"CCL persistent RS buffers allocated out={out_shape} inter={inter_shape}")
        return [inter, out]


def cp_degree(mesh_config):
    """Number of context-parallel ranks in the Galaxy mesh."""
    return mesh_config.prefill.sp


def ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config=None):
    """All-reduce across TP devices.

    Sync ``ttnn.all_reduce`` by default. With ``GEMMA4_CCL_ASYNC=1``, uses
    reduce_scatter_minimal_async + all_gather_async (tt_transformers composite
    pattern) on ``ccl_manager.topology`` (Ring on P150x8).
    """
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis
    topology = ccl_manager.topology

    chunks = ccl_chunks_per_sync()
    workers = ccl_num_workers_per_link()
    nbuf = ccl_num_buffers_per_channel()
    if ccl_async_enabled():
        tp = mesh_config.tp
        rs_bufs = ccl_manager.get_persistent_rs_buffers(tensor, memory_config, tp)
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=rs_bufs,
            dim=3,
            multi_device_global_semaphore=ccl_manager.get_rs_semaphore(),
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
            num_links=ccl_manager.num_links,
            cluster_axis=tp_axis,
            memory_config=memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            chunks_per_sync=chunks,
            num_workers_per_link=workers,
            num_buffers_per_channel=nbuf,
        )
        tensor.deallocate(True)
        # Do not pass a persistent AG buffer: the gather result is returned as a
        # normal activation and force-deallocated by callers. Persistent RS out
        # aliases ``scattered`` when rs_bufs is set — do not free it either.
        gathered = ttnn.experimental.all_gather_async(
            scattered,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
            num_links=ccl_manager.num_links,
            cluster_axis=tp_axis,
            topology=topology,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
            chunks_per_sync=chunks,
            num_workers_per_link=workers,
            num_buffers_per_channel=nbuf,
        )
        if rs_bufs is None:
            scattered.deallocate(True)
        return gathered

    # Sync all_reduce: omit deprecated num_links/topology (Sep-2026 removal);
    # Fabric / cluster_axis supply those defaults (same as sync all_gather).
    result = ttnn.all_reduce(
        tensor,
        cluster_axis=tp_axis,
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    return result


def ccl_allgather(tensor, mesh_config, ccl_manager, dim=3, memory_config=None):
    """All-gather across TP devices."""
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis
    topology = ccl_manager.topology
    chunks = ccl_chunks_per_sync()
    workers = ccl_num_workers_per_link()
    nbuf = ccl_num_buffers_per_channel()

    if ccl_async_enabled():
        # Fresh AG output each call (caller-owned); see ccl_allreduce note.
        gathered = ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
            num_links=ccl_manager.num_links,
            cluster_axis=tp_axis,
            topology=topology,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
            chunks_per_sync=chunks,
            num_workers_per_link=workers,
            num_buffers_per_channel=nbuf,
        )
        tensor.deallocate(True)
        return gathered

    # Sync all_gather: do not pass deprecated num_links/topology/chunks_* —
    # Fabric config supplies those; passing them only emits Sep-2026 warnings.
    gathered = ttnn.all_gather(
        tensor,
        dim=dim,
        cluster_axis=tp_axis,
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    return gathered
