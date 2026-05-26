# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""All-reduce that leaves activations replicated across the TP axis (reduce_scatter + all_gather)."""

from __future__ import annotations

from typing import Optional

import ttnn

__all__ = ["all_reduce_replicate"]


def all_reduce_replicate(
    tensor: ttnn.Tensor,
    *,
    mesh_device,
    tt_ccl,
    dim: int,
    cluster_axis: int = 1,
    topology: ttnn.Topology = ttnn.Topology.Linear,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """Sum ``tensor`` across all mesh devices and leave the result replicated.

    Implementation = ``reduce_scatter`` (across all devices, along ``dim``) followed by
    ``all_gather`` (along ``dim``). On a single-device mesh this is a no-op.

    ``memory_config`` controls the all-gather output placement. When ``None``, matches
    ``tensor``'s memory config (same pattern as ``tt_transformers.tt.ccl.tt_all_reduce``).
    """
    mesh_shape = list(mesh_device.shape)
    if mesh_shape[0] * mesh_shape[1] <= 1:
        return tensor

    output_mem = memory_config if memory_config is not None else tensor.memory_config()
    num_links = tt_ccl.get_num_links(cluster_axis)

    scattered = ttnn.experimental.reduce_scatter_minimal_async(
        tensor,
        persistent_output_buffers=None,
        dim=dim,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        cluster_axis=cluster_axis,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )
    tensor.deallocate(True)

    gathered = ttnn.experimental.all_gather_async(
        scattered,
        persistent_output_buffer=None,
        dim=dim,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=num_links,
        topology=topology,
        memory_config=output_mem,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        cluster_axis=cluster_axis,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )
    scattered.deallocate(True)
    return gathered
