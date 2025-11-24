# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import ttnn


def test_tensor_accessor_args(device):
    shape = [3, 128, 160]
    shard_shape = [3, 128, 32]
    py_tensor = torch.rand(shape).to(torch.bfloat16)
    grid_size = device.compute_with_storage_grid_size()
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))
    grid = ttnn.CoreRangeSet([core_range])
    nd_shard_spec = ttnn.NdShardSpec(shape, grid)

    sharded_memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, nd_shard_spec)
    tt_tensor_sharded = ttnn.from_torch(
        py_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_memory_config
    )

    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG
    tt_tensor_dram = ttnn.from_torch(
        py_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_memory_config
    )

    sharded_accessor = ttnn.TensorAccessorArgs(tt_tensor_sharded)
    dram_accessor = ttnn.TensorAccessorArgs(tt_tensor_dram)

    sharded_accessor_cta = sharded_accessor.get_compile_time_args()
    sharded_accessor_crta = sharded_accessor.get_common_runtime_args()
    print(sharded_accessor_cta)
    print(sharded_accessor_crta)

    dram_accessor_cta = dram_accessor.get_compile_time_args()
    dram_accessor_crta = dram_accessor.get_common_runtime_args()
    assert len(dram_accessor_cta) == 1
    assert len(dram_accessor_crta) == 0
