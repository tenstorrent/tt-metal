# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Matmul Micro Op Test - Single Core
Tests matmul operation with shape [1, 7K] x [7K, 32]
All tensors on a single core:
- Input A (in0): 1x7K, HEIGHT_SHARDED on single core
- Input B (in1): 7Kx32, WIDTH_SHARDED on single core
- Output: 1x32, WIDTH_SHARDED on single core
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.matmul.op import MatmulSingleCore


@pytest.mark.parametrize(
    "M, K, N",
    [
        (1, 7168, 32),  # Single core: 1x7K x 7Kx32 -> 1x32
    ],
)
def test_matmul_single_core(device, M, K, N):
    """Test single-core matmul operation with fully sharded inputs"""

    # Tile dimensions
    tile_height = 1  # Tiny tile height for A and output
    tile_width = 32
    b_tile_height = 32  # Standard tile height for B
    b_tile_width = 32

    # Single core
    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRange(core, core)
    core_grid = ttnn.CoreGrid(x=1, y=1)

    # Calculate tiles
    num_tiles_m = M // tile_height
    num_tiles_k = K // tile_width
    num_tiles_n = N // tile_width

    logger.info(f"Testing single-core matmul with shape [{M}, {K}] x [{K}, {N}]")
    logger.info(f"Tiles: M={num_tiles_m}, K={num_tiles_k}, N={num_tiles_n}")

    # Create PyTorch tensors for reference
    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_a.float(), torch_b.float()).bfloat16()

    # Create HEIGHT_SHARDED memory config for input A
    # Single core has full 1xK tensor
    input_a_shard_shape = (M, K)
    input_a_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({core_range}),
        input_a_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_a_shard_spec
    )

    # Create input A (height-sharded on single core)
    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_a_mem_config,
        tile=ttnn.Tile([tile_height, tile_width]),
    )

    logger.info(f"Created input A with shard shape {input_a_shard_shape}")

    # Create WIDTH_SHARDED memory config for input B
    # Single core has full KxN tensor
    input_b_shard_shape = (K, N)
    input_b_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({core_range}),
        input_b_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_b_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_b_shard_spec
    )

    # Create input B (width-sharded on single core)
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_b_mem_config,
        tile=ttnn.Tile([b_tile_height, b_tile_width]),
    )

    logger.info(f"Created input B with shard shape {input_b_shard_shape}")

    # Create WIDTH_SHARDED memory config for output
    # Single core produces full MxN output
    output_shard_shape = (M, N)
    output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({core_range}),
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Create output tensor
    torch_output_zeros = torch.zeros((M, N), dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=ttnn.Tile([tile_height, tile_width]),
    )

    logger.info(f"Created output tensor with shard shape {output_shard_shape}")

    # Run matmul operation
    logger.info("Running matmul operation...")
    ttnn_result = MatmulSingleCore.op(
        ttnn_a,
        ttnn_b,
        ttnn_output,
        core_grid,
        tile_height=tile_height,
        tile_width=tile_width,
        fp32_dest_acc_en=False,
    )

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == (M, N), f"Expected shape ({M}, {N}), got {output_torch.shape}"

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

    logger.info("✓ Single-core matmul test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
