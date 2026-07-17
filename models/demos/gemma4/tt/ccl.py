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
    """
    return 2 if is_blackhole() else 1


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
            f"topology={topo_name} async={int(ccl_async_enabled())}"
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

    if ccl_async_enabled():
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=ccl_manager.get_rs_semaphore(),
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
            num_links=ccl_manager.num_links,
            cluster_axis=tp_axis,
            memory_config=memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        tensor.deallocate(True)
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
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
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

    if ccl_async_enabled():
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
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        tensor.deallocate(True)
        return gathered

    gathered = ttnn.all_gather(
        tensor,
        dim=dim,
        cluster_axis=tp_axis,
        num_links=ccl_manager.num_links,
        topology=topology,
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    return gathered
