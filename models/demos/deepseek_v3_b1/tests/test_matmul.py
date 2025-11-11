"""
TTNN Matrix Multiplication Test with Width Sharding and MCast 1D
Tests matmul operation with shape [32, 7168] x [7168, 1536]
Input 0 is width-sharded across 56 cores using ttnn.experimental.deepseek_b1.matmul_1d
"""

import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_matmul(device):
    """Test TTNN matmul with width-sharded input and mcast 1d using DeepSeek B1 op"""

    # Hardcoded matrix dimensions
    tile_height = 1
    m = tile_height  # set to minimum tile height for b1
    k = 7168
    n = 1536

    # Core grid: 48 cores (6x8 grid)
    grid_size = ttnn.CoreGrid(y=8, x=6)
    num_cores = grid_size.num_cores

    logger.info(f"Testing matmul with shape [{m}, {k}] x [{k}, {n}]")
    logger.info(f"Using {num_cores} cores in {grid_size.y}x{grid_size.x} grid")

    # Create PyTorch tensors for reference
    torch.manual_seed(0)
    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((k, n), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_a.float(), torch_b.float()).bfloat16()

    # Create width-sharded memory config for input A
    # Width sharding distributes along the K dimension (7168)
    # Each core gets k/num_cores width
    k_tiles = k // 32  # 224 tiles
    k_per_core_tiles = k_tiles  # Full shard on single core

    shard_shape = (m, k_per_core_tiles * 32)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}),
        shard_shape,
        ttnn.ShardOrientation.COL_MAJOR,
    )

    width_sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    # Create width-sharded input A directly
    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=width_sharded_mem_config,
        tile=ttnn.Tile([tile_height, 32]),
    )

    # Create interleaved input B
    ttnn_b = ttnn.from_torch(
        torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Calculate matmul parameters
    m_tiles = m // tile_height  # 1 tile
    n_tiles = n // 32  # 48 tiles

    logger.info(f"Matmul params: in0_block_w={k_per_core_tiles}, per_core_M={m_tiles}, per_core_N=1")

    # Create width-sharded memory config for output
    # Each core produces per_core_N tiles in the N dimension
    per_core_N = 1
    output_shard_shape = (m, per_core_N * 32)
    output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}),
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_width_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec
    )

    # Use the new DeepSeek B1 matmul_1d operation
    print(f"ttnn_a: {ttnn_a}")
    ttnn_output = ttnn.experimental.deepseek_b1.matmul_1d(
        ttnn_a,
        ttnn_b,
        core_grid=grid_size,
        in0_block_w=k_per_core_tiles,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=m_tiles,
        per_core_N=per_core_N,
        fuse_batch=True,
        mcast_in0=True,
        memory_config=output_width_sharded_mem_config,
    )

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(ttnn_output)

    # Verify output shape
    assert output_torch.shape == (m, n), f"Expected shape ({m}, {n}), got {output_torch.shape}"

    # Compute PCC (Pearson Correlation Coefficient) for accuracy check
    output_flat = output_torch.flatten().float()
    expected_flat = torch_output.flatten().float()

    mean_output = output_flat.mean()
    mean_expected = expected_flat.mean()

    output_centered = output_flat - mean_output
    expected_centered = expected_flat - mean_expected

    correlation = (output_centered * expected_centered).sum()
    norm_output = torch.sqrt((output_centered**2).sum())
    norm_expected = torch.sqrt((expected_centered**2).sum())

    pcc = correlation / (norm_output * norm_expected)

    logger.info(f"PCC: {pcc.item():.6f}")

    # Check that PCC is above threshold
    assert pcc.item() > 0.99, f"PCC {pcc.item():.6f} is below threshold 0.99"

    logger.info("✓ Width-sharded mcast 1d matmul test passed using ttnn.experimental.deepseek_b1.matmul_1d!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
