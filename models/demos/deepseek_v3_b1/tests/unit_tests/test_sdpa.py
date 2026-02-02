# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN SDPA S*V Test

Tests the SDPA attention output computation: S @ V
where V is a subset of K, with v_width tiles read from K.

Input tensors are sharded on a single core.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.sdpa.op import SdpaSingleCore


@pytest.mark.parametrize(
    "num_tiles_k, num_tiles_v, chunk_size",
    [
        (6, 4, 2),
    ],
)
def test_sdpa(device, num_tiles_k, num_tiles_v, chunk_size):
    """Test TTNN SDPA S@V operation on a single core"""

    # Tile dimensions
    tile_height = 8
    tile_width = 32
    s_tile = ttnn.Tile([tile_height, tile_width])
    k_tile = ttnn.Tile([32, 32])

    # S shape: [1 tile height, num_tiles_k tiles width] = [32, num_tiles_k * 32]
    # K shape: [chunk_size tiles height, num_tiles_k tiles width]
    # Output shape: [1 tile height, num_tiles_v tiles width]
    s_shape = (tile_height, chunk_size * tile_width)
    k_shape = (chunk_size * tile_width, num_tiles_k * tile_width)
    out_shape = (tile_height, num_tiles_v * tile_width)

    logger.info(f"S shape: {s_shape}, K shape: {k_shape}, Output shape: {out_shape}")

    # Create input PyTorch tensors
    torch.manual_seed(42)

    # S is attention scores, K contains V as a subset
    torch_s = torch.randn(s_shape, dtype=torch.bfloat16) * 0.1
    torch_k = torch.randn(k_shape, dtype=torch.bfloat16) * 0.1

    # Compute reference output using PyTorch: S @ V (where V is subset of K)
    torch_out_expected = SdpaSingleCore.golden(torch_s, torch_k, num_tiles_v * tile_width)

    logger.info(f"Expected output shape: {torch_out_expected.shape}")

    # Shard spec: single core
    s_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        s_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    s_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, s_shard_spec)

    k_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        k_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    k_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, k_shard_spec)

    out_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        out_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard_spec)

    # Create input tensors sharded on single core
    ttnn_s = ttnn.from_torch(
        torch_s,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=s_mem_config,
        tile=s_tile,
    )
    ttnn_k = ttnn.from_torch(
        torch_k,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=k_mem_config,
        tile=k_tile,
    )

    # Create output tensor sharded on same core
    torch_out = torch.zeros(out_shape, dtype=torch.bfloat16)
    ttnn_out = ttnn.from_torch(
        torch_out,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem_config,
        tile=s_tile,
    )

    logger.info("Created tensors sharded on single core")

    # Run SDPA operation
    logger.info("Running SDPA S*V operation...")
    result = SdpaSingleCore.op(
        ttnn_s,
        ttnn_k,
        ttnn_out,
        chunk_size=chunk_size,
        num_tiles_k=num_tiles_k,
        num_tiles_v=num_tiles_v,
    )

    # Convert back to torch for verification
    out_torch = ttnn.to_torch(result)

    # Verify output shape
    assert out_torch.shape == out_shape, f"Expected output shape {out_shape}, got {out_torch.shape}"

    # Verify results
    logger.info("Verifying SDPA results...")

    # Check output
    max_diff = torch.max(torch.abs(out_torch - torch_out_expected)).item()
    mean_diff = torch.mean(torch.abs(out_torch - torch_out_expected)).item()
    logger.info(f"Max absolute difference: {max_diff}")
    logger.info(f"Mean absolute difference: {mean_diff}")

    passing, pcc_message = comp_pcc(torch_out_expected, out_torch, 0.99)
    logger.info(f"{pcc_message}")

    assert passing, f"Output PCC failed: {pcc_message}"

    logger.info("✓ SDPA S*V test passed!")
