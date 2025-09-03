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


class EncoderParallelConfig(NamedTuple):
    tensor_parallel: ParallelFactor


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
    # Until barrier_semahpore hang is fixed, sync devices before all-gather.
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
        num_workers_per_link=4,
        chunks_per_sync=80,
        num_buffers_per_channel=4,
    )
    # ttnn.synchronize_device(x.device())
    # reshape back to original expected shape
    if h != 1:
        x_g = x_g.reshape(b, h, w, -1)
    return x_g
