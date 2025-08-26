# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import ttnn


class ParallelFactor(NamedTuple):
    factor: int
    mesh_axis: int


class DiTParallelConfig(NamedTuple):
    cfg_parallel: ParallelFactor
    tensor_parallel: ParallelFactor
    sequence_parallel: ParallelFactor


class VAEParallelConfig(NamedTuple):
    tensor_parallel: ParallelFactor


class OldParallelConfig(NamedTuple):
    mesh_shape: tuple[int, int]
    factor: int
    mesh_axis: int


def vae_all_gather(ccl_manager, x: ttnn.Tensor, cluster_axis: int = 1, dim: int = 3) -> ttnn.Tensor:
    global_semaphores = ccl_manager.get_ag_ping_pong_semaphore(cluster_axis)
    barrier_semaphore = ccl_manager.get_barrier_semaphore(cluster_axis)

    # reshape to b,1,h*w,c. This was tested to be faster. Need to verify overhead. TODO: Cleanup
    b, h, w, c = x.shape
    if h != 1:  # Check if its already in desired shape. E.g group norm already reshaped to 1,1,h*w,c
        x = x.reshape(b, 1, h * w, c)

    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # All gather requires tile layout

    # NOTE: We can't use ping-pong persistent buffers because we run out of memory.
    # Single-buffered persistent buffers is a potential correctness issue, so we can't do that.
    # Using barrier_semaphore is a good solution, but right now it causes hangs when VAE integrated into pipeline.
    # Until barrier_semahpore hang is fixed, sync devices before and after all-gather.
    ttnn.synchronize_device(x.device())
    x_g = ttnn.experimental.all_gather_async(
        input_tensor=x,
        dim=dim,
        persistent_output_buffer=None,
        # barrier_semaphore=barrier_semaphore,
        multi_device_global_semaphore=global_semaphores,
        topology=ttnn.Topology.Linear,
        cluster_axis=cluster_axis,
        num_links=ccl_manager.num_links,
        # num_workers_per_link=4,
        # chunks_per_sync=80,
        # num_buffers_per_channel=4,
    )
    ttnn.synchronize_device(x.device())
    # reshape back to original expected shape
    if h != 1:
        x_g = x_g.reshape(b, h, w, -1)
    return x_g


# TODO: Simplify with CCLManager
class EncoderParallelManager:
    def __init__(
        self,
        mesh_device,
        topology,
        mesh_axis,
        num_links=1,
    ):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology

        self.tensor_parallel = OldParallelConfig(
            mesh_shape=tuple(mesh_device.shape), factor=mesh_device.shape[mesh_axis], mesh_axis=mesh_axis
        )

        # Setup semaphores
        self._init_subdevice()

        # SD35-specific semaphores
        semaphore_names = [("ping_pong", 2 * 2), ("rs_ping_pong", 3 * 2), ("ar_ping_pong", 3 * 2)]
        self._init_semaphores(semaphore_names)
        self.ping_pong_idx = 0
        self.rs_ping_pong_idx = 0
        self.ar_ping_pong_idx = 0

    def _init_subdevice(self):
        compute_grid_size = self.mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )

        _worker_sub_device = ttnn.SubDevice(
            [
                self.ccl_cores,
            ]
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)

    def _init_semaphores(self, semaphore_names):
        # Initialize semaphores for each necessary CCL, None if a single semaphore else length of semaphore list
        self.sems = {}
        for sem_name, len_sem in semaphore_names:
            n_sems = 1 if len_sem is None else len_sem
            sems = [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(n_sems)]
            if len_sem is None:
                sems = sems[0]
            self.sems[sem_name] = sems

    def get_ping_pong_semaphore(self):
        cur_idx = self.ping_pong_idx
        self.ping_pong_idx = 2 - self.ping_pong_idx
        return self.sems["ping_pong"][cur_idx : cur_idx + 2]

    def get_rs_ping_pong_semaphore(self):
        cur_idx = self.rs_ping_pong_idx
        n_sems = 3
        self.rs_ping_pong_idx = (cur_idx + 1) % 2
        return self.sems["rs_ping_pong"][cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_ar_ping_pong_semaphore(self):
        cur_idx = self.ar_ping_pong_idx
        n_sems = 3
        self.ar_ping_pong_idx = (cur_idx + 1) % 2
        return self.sems["ar_ping_pong"][cur_idx * n_sems : (cur_idx + 1) * n_sems]

    @property
    def is_tensor_parallel(self):
        return self.tensor_parallel.factor > 1
