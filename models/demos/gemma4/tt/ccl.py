# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from loguru import logger

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


def default_ccl_topology(mesh_device=None):
    """Default CCL topology for Gemma4 TP collectives.

    Override with ``GEMMA4_CCL_TOPOLOGY=ring|linear``.

    Policy (when env unset):
      * **Ring** only on **Blackhole** meshes with **≥8 devices** (P150x8 TTFT
        sweep: Ring+sync ~28.8s vs Linear+sync ~31.0s @ 31B/128k).
      * **Linear** everywhere else — including Wormhole T3K 1x8. Ring on WH
        drops 26B-A4B ``test_full_model`` PCC below the TEMP 0.76 gate
        (~0.7505 vs ~0.77/0.94 with Linear / main). Ring on 4-device BH also
        drops 12B full-model PCC (~0.97 → ~0.90).

    Async RS+AG is correct but slower than sync on P150x8 — keep
    ``GEMMA4_CCL_ASYNC=0`` unless re-swept.
    """
    override = os.environ.get("GEMMA4_CCL_TOPOLOGY", "").strip().lower()
    if override in ("ring", "r"):
        return ttnn.Topology.Ring
    if override in ("linear", "line", "l"):
        return ttnn.Topology.Linear

    n = mesh_device.get_num_devices() if mesh_device is not None else 0
    # Ring TTFT win was swept on BH P150x8 only. WH T3K is also n=8 but must
    # stay Linear for MoE PCC (matches main's hardcoded Linear all-reduce).
    if n:
        if n >= 8 and is_blackhole():
            return ttnn.Topology.Ring
        return ttnn.Topology.Linear

    try:
        cluster = ttnn.cluster.get_cluster_type()
    except Exception:
        cluster = None

    # No mesh_device: Ring only on full 8-device BH LoudBox / BH Galaxy.
    # Do not treat WH T3K / Galaxy cluster types as Ring defaults.
    ring_when_unknown_n = ()
    for name in ("P150_X8", "BLACKHOLE_GALAXY"):
        if hasattr(ttnn.cluster.ClusterType, name):
            ring_when_unknown_n += (getattr(ttnn.cluster.ClusterType, name),)
    if cluster in ring_when_unknown_n:
        return ttnn.Topology.Ring
    return ttnn.Topology.Linear


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
            topology = default_ccl_topology(mesh_device)
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
            self._ag_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(2)])
            self._barrier_semaphores.append(ttnn.create_global_semaphore(mesh_device, core_range_set, 0))
        ttnn.synchronize_device(mesh_device)

        self._rs_idx = 0
        self._ag_idx = 0
        self._barrier_idx = 0
        # shape_key -> ttnn.Tensor (DRAM interleaved zeros)
        self._persistent_ag: dict = {}
        self._persistent_rs_out: dict = {}
        self._persistent_rs_inter: dict = {}

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

    def _shape_key(self, shape, dtype, memory_config):
        return (tuple(int(x) for x in shape), str(dtype), str(memory_config))

    def _alloc_like(self, ref_tensor, memory_config):
        return ttnn.zeros_like(ref_tensor, device=self.mesh_device, memory_config=memory_config)

    def get_persistent_ag_buffer(self, scattered, memory_config, tp):
        """Allocate a persistent AG destination sized by TP group width.

        Disabled by default in ``ccl_allreduce`` / ``ccl_allgather``: the gathered
        result is returned as a normal activation and Gemma4 force-deallocates
        those, which would free a manager-cached buffer. Kept for opt-in / tests.
        """
        if not ccl_persistent_buffers_enabled():
            return None
        if tp <= 1:
            return None
        # All-gather expands dim=3 by the TP group size (cluster_axis width).
        out_shape = list(scattered.shape)
        out_shape[3] = int(out_shape[3]) * tp
        key = self._shape_key(out_shape, scattered.dtype, memory_config)
        buf = self._persistent_ag.get(key)
        if buf is None:
            buf = ttnn.zeros(
                out_shape,
                dtype=scattered.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=memory_config,
            )
            self._persistent_ag[key] = buf
            logger.debug(f"CCL persistent AG buffer allocated shape={out_shape}")
        return buf

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
