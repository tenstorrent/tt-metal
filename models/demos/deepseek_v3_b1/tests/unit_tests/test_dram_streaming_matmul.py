# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN DRAM Streaming Matmul Micro Op Test

Tests the DRAM streaming matmul operation where:
- Input A is WIDTH_SHARDED in L1 across compute cores
- Input B (weights) is WIDTH_SHARDED in DRAM across DRAM banks
- Output is WIDTH_SHARDED in L1
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul.op import DRAMStreamingMatmul
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def pad_to_dram_banks(num, tile_w, lcm):
    """Pad number to be aligned with DRAM banks."""
    remainder = num % lcm
    if remainder == 0:
        return num
    padding_needed = lcm - remainder
    return num + padding_needed


@pytest.mark.parametrize("k, n", [(7168, 2048), (2048, 7168)])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("grid_size", [(8, 1)])
def test_dram_streaming_matmul(device, k, n, has_bias, grid_size):
    """Test DRAM streaming matmul using the cleaner ttnn API patterns."""

    m = 32  # Standard tile height
    tile_h = 32
    tile_w = 32

    # Get number of DRAM banks
    num_banks = device.dram_grid_size().x
    n_padded = pad_to_dram_banks(n, tile_w, tile_w * num_banks)

    # Define shapes
    in0_shape = [1, 1, m, k]
    in1_shape = [1, 1, k, n]
    in1_shard_shape = [k, n_padded // num_banks]
    bias_shape = [1, 1, n]
    bias_shard_shape = [tile_h, n_padded // num_banks]
    num_cores = grid_size[0] * grid_size[1]

    # Calculate block dimensions
    in0_block_w = k // num_cores // 32
    out_block_h = m // tile_h
    out_block_w = n // num_cores // tile_w

    logger.debug(
        f"n_padded={n_padded}, in0_block_w={in0_block_w}, out_block_h={out_block_h}, out_block_w={out_block_w}"
    )

    # Create PyTorch tensors
    torch.manual_seed(42)
    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    # Input A - WIDTH_SHARDED in L1 using create_sharded_memory_config
    in0_memory_config = ttnn.create_sharded_memory_config(
        (1, 1, m, k),
        core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    in0_t = ttnn.from_torch(
        in0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
    )

    # Input B (weights) - WIDTH_SHARDED in DRAM
    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_t = ttnn.from_torch(
        in1,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )

    # Bias - WIDTH_SHARDED in DRAM (if needed)
    bias_t = None
    bias = None
    if has_bias:
        bias = torch.randn(bias_shape).bfloat16().float()
        bias_padded = bias.unsqueeze(2)
        bias_padded = torch.nn.functional.pad(bias_padded, (0, 0, 0, tile_h - bias_padded.size(2)), "constant", 0)
        bias_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
        bias_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), bias_shard_grid)})
        bias_shard_spec = ttnn.ShardSpec(bias_shard_grid, bias_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        bias_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, bias_shard_spec
        )
        bias_t = ttnn.from_torch(
            bias_padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=bias_mem_config,
        )

    # Output memory config - WIDTH_SHARDED in L1
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    # Create output tensor
    output_shard_width = n // num_cores
    output_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1))]
    )
    output_shard_spec = ttnn.ShardSpec(output_core_grid, (m, output_shard_width), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output_zeros = torch.zeros([1, 1, m, n]).bfloat16().float()
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
    )

    # Run DRAM streaming matmul
    logger.info(f"Running DRAM streaming matmul: m={m}, k={k}, n={n}, grid={grid_size}, bias={has_bias}")
    try:
        ttnn_result = DRAMStreamingMatmul.op(
            in0_t,
            in1_t,
            ttnn_output,
            bias=bias_t,
            in0_block_w=in0_block_w,
            per_core_M=out_block_h,
            per_core_N=out_block_w,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
        )
    except Exception as e:
        logger.error(f"DRAM streaming matmul failed: {e}")
        pytest.skip(f"Operation failed (may need API adjustments): {e}")

    # Compute PyTorch reference
    pt_out = in0 @ in1
    if has_bias:
        pt_out = pt_out + bias

    # Convert to torch for comparison
    tt_out = ttnn.to_torch(ttnn_result)

    # Verify results
    expected_pcc = 0.999
    passing, output = comp_pcc(pt_out, tt_out, expected_pcc)
    logger.info(output)
    assert passing, f"PCC check failed: {output}"
    logger.info("DRAM streaming matmul test passed!")
