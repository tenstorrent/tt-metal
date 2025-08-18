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


class EncoderParallelConfig(NamedTuple):
    tensor_parallel: ParallelFactor


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
