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
