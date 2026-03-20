# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE


@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_slow_mul_op(device):
    # Create random inputs with shape [batch, num_heads, seq_len, head_dim]
    torch_in0 = torch.randn(1, 1, 4096, 2560, dtype=torch.float32)
    torch_in1 = torch.randn(1, 1, 4096, 2560, dtype=torch.float32)

    # First create tensors in interleaved memory
    tt_in0 = ttnn.from_torch(
        torch_in0,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in1 = ttnn.from_torch(
        torch_in1,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create block sharded memory config matching geglu (core_grid=10x10, per_core_M=13, per_core_N=8)
    core_range = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(9, 9),
    )
    core_range_set = ttnn.CoreRangeSet({core_range})
    shard_shape = [416, 256]  # 13 tiles * 32, 8 tiles * 32
    shard_spec = ttnn.ShardSpec(
        core_range_set,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    block_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )

    # Reshard to block sharded L1 (this will handle padding automatically)
    tt_in0 = ttnn.to_memory_config(tt_in0, block_sharded_mem_config)
    tt_in1 = ttnn.to_memory_config(tt_in1, block_sharded_mem_config)

    # Multiply
    tt_out = ttnn.mul_(tt_in0, tt_in1, use_legacy=False, fast_and_approximate_mode=True)

    # Convert back to torch to verify output shape
    output_torch = ttnn.to_torch(tt_out)


@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_fast_mul_op(device):
    # Create random inputs with shape [batch, num_heads, seq_len, head_dim]
    torch_in0 = torch.randn(1, 1, 4096, 2560, dtype=torch.float32)
    torch_in1 = torch.randn(1, 1, 4096, 2560, dtype=torch.float32)

    # First create tensors in interleaved memory
    tt_in0 = ttnn.from_torch(
        torch_in0,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in1 = ttnn.from_torch(
        torch_in1,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create block sharded memory config matching geglu (core_grid=10x8, per_core_M=16, per_core_N=8)
    core_range = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(9, 7),
    )
    core_range_set = ttnn.CoreRangeSet({core_range})
    shard_shape = [512, 256]  # 16 tiles * 32, 8 tiles * 32
    shard_spec = ttnn.ShardSpec(
        core_range_set,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    block_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )

    # Reshard to block sharded L1 (this will handle padding automatically)
    tt_in0 = ttnn.to_memory_config(tt_in0, block_sharded_mem_config)
    tt_in1 = ttnn.to_memory_config(tt_in1, block_sharded_mem_config)

    # Multiply
    tt_out = ttnn.mul_(tt_in0, tt_in1, use_legacy=False, fast_and_approximate_mode=True)

    # Convert back to torch to verify output shape
    output_torch = ttnn.to_torch(tt_out)
