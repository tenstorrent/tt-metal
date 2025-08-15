# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import ttnn
from .manager import CCLManager


class ParallelFactor(NamedTuple):
    factor: int
    mesh_axis: int


class DiTParallelConfig(NamedTuple):
    cfg_parallel: ParallelFactor
    tensor_parallel: ParallelFactor
    sequence_parallel: ParallelFactor


class OldParallelConfig(NamedTuple):
    mesh_shape: tuple[int, int]
    factor: int
    mesh_axis: int


# TODO: Simplify with CCLManager
class VAEParallelManager:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        new_gather_handles: list[ttnn._ttnn.global_semaphore.global_sempahore],
        gather_semaphore: ttnn._ttnn.global_semaphore.global_sempahore,
        reduce_from_semaphore: ttnn._ttnn.global_semaphore.global_sempahore,
        reduce_to_semaphore: ttnn._ttnn.global_semaphore.global_sempahore,
        num_links: int,
        barrier_semaphore: list[ttnn._ttnn.global_semaphore.global_sempahore],
    ):
        self.device = device
        self.new_gather_handles = new_gather_handles
        self.barrier_semaphore = barrier_semaphore
        self.gather_semaphore = gather_semaphore
        self.reduce_from_semaphore = reduce_from_semaphore
        self.reduce_to_semaphore = reduce_to_semaphore
        self.num_links = num_links
        self.ping_pong_idx = 0

    def vae_all_gather(self, x: ttnn.Tensor) -> ttnn.Tensor:
        semaphores = self.new_gather_handles[self.ping_pong_idx * 2 : (self.ping_pong_idx + 1) * 2]
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # All gather requires tile layout
        x_g = ttnn.experimental.all_gather_async(
            input_tensor=x,
            dim=3,
            multi_device_global_semaphore=semaphores,
            topology=ttnn.Topology.Linear,
            mesh_device=self.device,
            cluster_axis=1,
            num_links=self.num_links,
            barrier_semaphore=self.barrier_semaphore[self.ping_pong_idx],
        )

        self.ping_pong_idx = 1 - self.ping_pong_idx
        return x_g


def create_vae_parallel_manager(
    vae_device: ttnn.MeshDevice,
    ccl_manager: CCLManager,
) -> VAEParallelManager:
    return VAEParallelManager(
        device=vae_device,
        new_gather_handles=[
            ttnn.create_global_semaphore(vae_device, ccl_manager.ccl_cores, 0) for _ in range(2 * 2)
        ],  # Ping pong
        barrier_semaphore=[
            ttnn.create_global_semaphore(vae_device, ccl_manager.ccl_cores, 0) for _ in range(2)
        ],  # Barrier
        gather_semaphore=ttnn.create_global_semaphore(vae_device, ccl_manager.ccl_cores, 0),
        reduce_from_semaphore=ttnn.create_global_semaphore(vae_device, ccl_manager.ccl_cores, 0),
        reduce_to_semaphore=ttnn.create_global_semaphore(vae_device, ccl_manager.ccl_cores, 0),
        num_links=ccl_manager.num_links,
    )


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
