# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole


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
    """Default CCL topology — Ring on P150x8 / P150x4 / P300x2 (matches tt_transformers).

    Override with ``GEMMA4_CCL_TOPOLOGY=ring|linear``.

    Measured on 31B / P150x8 / unbounded chunk4k / 128k
    (``isl_sweep_logs/p150x8_bg_lb/ccl_prefill_ab.tsv``): Ring+sync TTFT
    ~28.8s vs Linear+sync ~31.0s (~7% win). Async RS+AG is correct but slower
    than sync on this board — keep ``GEMMA4_CCL_ASYNC=0`` unless re-swept.
    """
    override = os.environ.get("GEMMA4_CCL_TOPOLOGY", "").strip().lower()
    if override in ("ring", "r"):
        return ttnn.Topology.Ring
    if override in ("linear", "line", "l"):
        return ttnn.Topology.Linear

    try:
        cluster = ttnn.cluster.get_cluster_type()
    except Exception:
        cluster = None

    ring_clusters = ()
    for name in ("P150_X8", "P150_X4", "P300_X2", "T3K", "GALAXY", "TG", "BLACKHOLE_GALAXY"):
        if hasattr(ttnn.cluster.ClusterType, name):
            ring_clusters += (getattr(ttnn.cluster.ClusterType, name),)

    if cluster in ring_clusters:
        # Submeshes smaller than 8 devices on Galaxy/T3K fall back to Linear
        # (ring needs a closed loop); P150x4/P300x2 are ring-capable as-is.
        n = mesh_device.get_num_devices() if mesh_device is not None else 0
        if cluster in (
            getattr(ttnn.cluster.ClusterType, "T3K", None),
            getattr(ttnn.cluster.ClusterType, "GALAXY", None),
            getattr(ttnn.cluster.ClusterType, "TG", None),
            getattr(ttnn.cluster.ClusterType, "BLACKHOLE_GALAXY", None),
        ):
            if n and n < 8:
                return ttnn.Topology.Linear
        return ttnn.Topology.Ring

    if mesh_device is not None and mesh_device.get_num_devices() > 1:
        return ttnn.Topology.Linear
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

    result = ttnn.all_reduce(
        tensor,
        cluster_axis=tp_axis,
        num_links=ccl_manager.num_links,
        topology=topology,
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
