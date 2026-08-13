# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma-owned Gemma-4 CCL helpers with pre-created semaphores.

The shared Gemma-4 helpers route collectives through plain ``ttnn.all_reduce``
and ``ttnn.all_gather``. On a program-cache miss those operations create global
semaphores, whose setup performs a blocking command-queue drain. During trace
capture commands are recorded rather than executed, so that drain can never
finish and the process deadlocks.

DiffusionGemma instead decomposes all-reduce into the experimental minimal
reduce-scatter and all-gather operations. Both receive semaphores created by
``CCLManager`` before trace capture, keeping semaphore creation and blocking
writes off captured and post-capture execution paths.

``all_gather_async`` has one additional trap: ROW_MAJOR inputs and TILE inputs
padded on the gather dimension are automatically routed through
``composite_all_gather``, which is implemented with ``all_broadcast`` and ignores
the supplied semaphores. ``ccl_allgather`` therefore rejects those layouts.
Callers must tilize and tile-align the gather dimension first.
"""

import ttnn
from models.common.utility_functions import is_blackhole


def default_num_links():
    """Default TP-collective link count for the current architecture."""
    return 2 if is_blackhole() else 1


SEMAPHORE_BUFFER_DEPTH = 2
"""Number of pre-created semaphore sets to round-robin through.

DiffusionGemma uses one command queue, so programs execute in issue order and
double buffering provides one full collective of reuse slack.
"""


class CCLManager:
    """Own the mesh and pre-created semaphores used by DG collectives."""

    def __init__(self, mesh_device, num_links=None, topology=ttnn.Topology.Linear, buffer_depth=None):
        if num_links is None:
            num_links = default_num_links()
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
        self.num_devices = mesh_device.get_num_devices()

        grid = mesh_device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y
        core_range_set = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)

        depth = SEMAPHORE_BUFFER_DEPTH if buffer_depth is None else int(buffer_depth)
        if depth < 1:
            raise ValueError(f"buffer_depth must be >= 1, got {depth}")
        self._rs_semaphores = []
        self._ag_semaphores = []
        self._barrier_semaphores = []
        for _ in range(depth):
            self._rs_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(3)])
            self._ag_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(2)])
            self._barrier_semaphores.append(ttnn.create_global_semaphore(mesh_device, core_range_set, 0))
        ttnn.synchronize_device(mesh_device)

        self._depth = depth
        self._rs_idx = 0
        self._ag_idx = 0
        self._barrier_idx = 0

    def get_rs_semaphore(self):
        """Return three reduce-scatter semaphores and advance the ring."""
        semaphores = self._rs_semaphores[self._rs_idx]
        self._rs_idx = (self._rs_idx + 1) % self._depth
        return semaphores

    def get_ag_semaphore(self):
        """Return two all-gather semaphores and advance the ring."""
        semaphores = self._ag_semaphores[self._ag_idx]
        self._ag_idx = (self._ag_idx + 1) % self._depth
        return semaphores

    def get_barrier_semaphore(self):
        """Return one barrier semaphore and advance the ring."""
        semaphore = self._barrier_semaphores[self._barrier_idx]
        self._barrier_idx = (self._barrier_idx + 1) % self._depth
        return semaphore


def ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config=None):
    """All-reduce across TP devices without creating semaphores."""
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis

    scattered = ttnn.experimental.reduce_scatter_minimal_async(
        tensor,
        dim=3,
        cluster_axis=tp_axis,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        multi_device_global_semaphore=ccl_manager.get_rs_semaphore(),
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    gathered = ttnn.experimental.all_gather_async(
        scattered,
        dim=3,
        cluster_axis=tp_axis,
        mesh_device=ccl_manager.mesh_device,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        memory_config=memory_config,
    )
    scattered.deallocate(True)
    return gathered


def apply_allreduce(tensor, mesh_config, ccl_manager, hidden_size: int):
    """DG-local replacement for Gemma-4 attention's ``apply_allreduce``."""
    return ccl_allreduce(tensor, mesh_config, ccl_manager)


def _validate_minimal_allgather_input(tensor, dim: int):
    rank = len(tensor.shape)
    gather_dim = dim if dim >= 0 else rank + dim
    if gather_dim < 0 or gather_dim >= rank:
        raise ValueError(f"invalid all-gather dim {dim} for rank-{rank} tensor")
    if tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(
            "DG ccl_allgather requires TILE_LAYOUT; ROW_MAJOR silently selects "
            "composite all_broadcast and bypasses the supplied semaphores"
        )
    if gather_dim == rank - 1 and int(tensor.shape[gather_dim]) % ttnn.TILE_SIZE != 0:
        raise ValueError(f"DG ccl_allgather dim {gather_dim} must be tile-aligned; got shape {tensor.shape}")
    if gather_dim == rank - 2 and int(tensor.shape[gather_dim]) % ttnn.TILE_SIZE != 0:
        raise ValueError(f"DG ccl_allgather dim {gather_dim} must be tile-aligned; got shape {tensor.shape}")


def ccl_allgather(tensor, mesh_config, ccl_manager, dim=3, memory_config=None):
    """All-gather across TP devices using caller-owned semaphores."""
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    _validate_minimal_allgather_input(tensor, dim)
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis

    gathered = ttnn.experimental.all_gather_async(
        tensor,
        dim=dim,
        cluster_axis=tp_axis,
        mesh_device=ccl_manager.mesh_device,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    return gathered
