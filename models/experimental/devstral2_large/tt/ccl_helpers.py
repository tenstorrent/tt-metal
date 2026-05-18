# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CCL helpers for the Devstral-2 / Ministral3 TT rewrite.

We deliberately keep activations **replicated** across the TP axis between layers so the per-module
boundary is symmetric with the HF reference and RMSNorm stays a local (non-distributed) op.

``tt_transformers.tt.ccl.tt_all_reduce`` on a 1xN mesh is misleadingly named: it performs only
``reduce_scatter``, leaving the result width-scattered. This helper closes the gap with a single
follow-up ``all_gather`` so callers get a fully reduced + replicated tensor.
"""

from __future__ import annotations

import ttnn

__all__ = ["all_reduce_replicate"]


def all_reduce_replicate(
    tensor: ttnn.Tensor,
    *,
    mesh_device,
    tt_ccl,
    dim: int,
    topology: ttnn.Topology = ttnn.Topology.Linear,
) -> ttnn.Tensor:
    """Sum ``tensor`` across all mesh devices and leave the result replicated.

    Implementation = ``reduce_scatter`` (across all devices, along ``dim``) followed by
    ``all_gather`` (along ``dim``). On a single-device mesh this is a no-op.
    """
    mesh_shape = list(mesh_device.shape)
    if mesh_shape[0] * mesh_shape[1] <= 1:
        return tensor

    num_links = tt_ccl.get_num_links()

    scattered = ttnn.experimental.reduce_scatter_minimal_async(
        tensor,
        persistent_output_buffers=None,
        dim=dim,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        dim=dim,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=num_links,
        topology=topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )
    scattered.deallocate(True)
    return gathered
