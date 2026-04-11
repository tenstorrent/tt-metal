# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn


class CCLManager:
    """CCL manager for Gemma4 tensor parallelism.

    Stores mesh_device reference and num_links for CCL operations.
    Semaphores are retained for the experimental async CCL path (see TODO below).
    """

    def __init__(self, mesh_device, num_links=1, topology=ttnn.Topology.Linear):
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
