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
        gather_semaphores: list[ttnn._ttnn.global_semaphore.global_sempahore],
        num_links: int,
    ):
        self.device = device
        self.gather_semaphores = gather_semaphores
        self.num_links = num_links
        self.ping_pong_idx = 0

        self.buffer_count = 1  # Double buffer causing OOM
        vae_shapes = [
            # [1, 128, 128, 512],
            # [1, 256, 256, 512],
            # [1, 512, 512, 512],
            # [1, 512, 512, 256],
            # [1, 1024, 1024, 256],
            # [1, 1024, 1024, 128],
            # more optimal reahaped versions
            [1, 1, 128 * 128, 512],
            [1, 1, 256 * 256, 512],
            [1, 1, 512 * 512, 512],
            [1, 1, 512 * 512, 256],
            [1, 1, 1024 * 1024, 256],
            [1, 1, 1024 * 1024, 128],
        ]

        # We only need to create buffers for what is used. Make this more creating and saving buffers as needed.
        self.vae_persistent_buffers = {}
        for buffer_shape in vae_shapes:
            self.vae_persistent_buffers[f"vae_all_gather_{buffer_shape}"] = [
                ttnn.zeros(
                    buffer_shape, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
                )  # Will be repplicated by default
                for _ in range(self.buffer_count)
            ]  # double buffer , depending on buffer_count

    def vae_all_gather(self, x: ttnn.Tensor, cluster_axis: int = 1, dim: int = 3) -> ttnn.Tensor:
        semaphores = self.gather_semaphores[self.ping_pong_idx * 2 : (self.ping_pong_idx + 1) * 2]

        # reshape to b,1,h*w,c. This was tested to be faster. Need to verify overhead. TODO: Cleanup
        b, h, w, c = x.shape
        if h != 1:  # Check if its already in desired shape. E.g group norm already reshaped to 1,1,h*w,c
            x = x.reshape(b, 1, h * w, c)

        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # All gather requires tile layout

        gather_shape = list(x.shape)
        gather_shape[dim] *= self.device.shape[cluster_axis]  # get_num_devices()

        x_g = ttnn.experimental.all_gather_async(
            input_tensor=x,
            dim=dim,
            persistent_output_buffer=self.vae_persistent_buffers[f"vae_all_gather_{gather_shape}"][
                self.ping_pong_idx % self.buffer_count
            ],
            multi_device_global_semaphore=semaphores,
            topology=ttnn.Topology.Linear,
            cluster_axis=cluster_axis,
            num_links=self.num_links,
            num_workers_per_link=4,
            chunks_per_sync=80,
            num_buffers_per_channel=4,
        )

        # reshape back to original expected shape
        if h != 1:
            x_g = x_g.reshape(b, h, w, -1)

        self.ping_pong_idx = 1 - self.ping_pong_idx
        return x_g


def create_vae_parallel_manager(
    vae_device: ttnn.MeshDevice,
    ccl_manager: CCLManager,
) -> VAEParallelManager:
    return VAEParallelManager(
        device=vae_device,
        gather_semaphores=[
            ttnn.create_global_semaphore(vae_device, ccl_manager.ccl_cores, 0) for _ in range(2 * 2)
        ],  # Ping pong
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
