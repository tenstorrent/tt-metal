# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Vision mesh sum-allreduce: width-sharded RS + AG (replaces L1 interleaved all_gather + fast_reduce).

from __future__ import annotations

import math
from typing import Any, Optional

import ttnn

from models.common.utility_functions import nearest_32
from models.experimental.devstarl2_small.devstral_utils.dram_sharded_matmul import (
    TILE,
    width_sharded_l1_memcfg,
)


def _vision_ccl_core_grid(configuration: Any) -> tuple[int, int]:
    grid = configuration.max_grid_size
    if hasattr(grid, "x") and hasattr(grid, "y"):
        return int(grid.x), int(grid.y)
    if isinstance(grid, (tuple, list)) and len(grid) >= 2:
        return int(grid[0]), int(grid[1])
    return 8, 8


def vision_ccl_width_sharded_memcfg(
    seq_len: int,
    feature_dim: int,
    configuration: Any,
) -> Optional[ttnn.MemoryConfig]:
    """Width-sharded L1 for vision CCL (seq × feature activations)."""
    num_cores_x, num_cores_y = _vision_ccl_core_grid(configuration)
    num_cores = num_cores_x * num_cores_y
    m_tiles = math.ceil(int(seq_len) / TILE)
    n_tiles = int(feature_dim) // TILE
    if n_tiles <= 0 or n_tiles % num_cores != 0:
        return None
    return width_sharded_l1_memcfg(m_tiles, n_tiles, num_cores_x, num_cores_y)


def vision_ccl_rs_width_sharded_memcfg(
    seq_len: int,
    feature_dim: int,
    num_devices: int,
    configuration: Any,
) -> Optional[ttnn.MemoryConfig]:
    """Width-sharded L1 for reduce-scatter output (feature / num_devices per chip)."""
    local_dim = int(feature_dim) // int(num_devices)
    local_dim = nearest_32(local_dim)
    return vision_ccl_width_sharded_memcfg(seq_len, local_dim, configuration)


def _prepare_ccl_input(
    tensor: ttnn.Tensor,
    sharded_mem: Optional[ttnn.MemoryConfig],
) -> ttnn.Tensor:
    if sharded_mem is not None:
        if tensor.is_sharded():
            if tensor.memory_config() != sharded_mem:
                return ttnn.to_memory_config(tensor, sharded_mem)
            return tensor
        return ttnn.interleaved_to_sharded(tensor, sharded_mem)
    if tensor.memory_config().buffer_type == ttnn.BufferType.L1:
        return ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
    return tensor


def vision_sum_all_reduce(
    partial: ttnn.Tensor,
    mesh_device,
    tt_ccl,
    seq_len: int,
    feature_dim: int,
    configuration: Any,
    *,
    cluster_axis: int = 1,
) -> ttnn.Tensor:
    """Sum partial matmul outputs across mesh (K-sharded wo/w2). RS+AG vs all_gather+reduce."""
    mesh_shape = list(mesh_device.shape)
    if mesh_shape == [1, 1] or (cluster_axis == 1 and 1 in mesh_shape and mesh_shape[1] == 1):
        return partial

    num_devices = int(mesh_shape[cluster_axis])
    rs_sharded_mem = vision_ccl_rs_width_sharded_memcfg(seq_len, feature_dim, num_devices, configuration)
    in_sharded_mem = vision_ccl_width_sharded_memcfg(seq_len, feature_dim, configuration)
    ag_out_mem = ttnn.DRAM_MEMORY_CONFIG

    ccl_in = _prepare_ccl_input(partial, in_sharded_mem)
    rs_out_mem = rs_sharded_mem if rs_sharded_mem is not None else ttnn.DRAM_MEMORY_CONFIG
    prepared = ccl_in is not partial

    scattered = ttnn.experimental.reduce_scatter_minimal_async(
        ccl_in,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        num_links=tt_ccl.get_num_links(cluster_axis),
        cluster_axis=cluster_axis,
        memory_config=rs_out_mem,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )
    if prepared:
        partial.deallocate(True)
    else:
        partial.deallocate(True)

    gathered = ttnn.experimental.all_gather_async(
        scattered,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=tt_ccl.get_num_links(cluster_axis),
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Linear,
        memory_config=ag_out_mem,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )
    scattered.deallocate(True)

    if (
        len(gathered.shape) == 4
        and int(gathered.shape[0]) == 1
        and int(gathered.shape[1]) == 1
        and int(gathered.shape[2]) == int(seq_len)
    ):
        return gathered
    return ttnn.reshape(gathered, [1, 1, int(seq_len), -1])


__all__ = [
    "vision_ccl_rs_width_sharded_memcfg",
    "vision_ccl_width_sharded_memcfg",
    "vision_sum_all_reduce",
]
