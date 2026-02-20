# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN SDPA Q*K^T Test

Tests the SDPA attention scores computation: QK^T = Q @ K^T
where Q and K are chunked and processed iteratively.

Input tensors are sharded on a single core.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.sdpa.op import SdpaSingleCore


@pytest.mark.parametrize(
    "num_tiles_k, num_tiles_v, chunk_size, num_chunks, scale",
    [
        (18, 16, 4, 2, 1),
        (18, 16, 4, 8, 0.5),
        (18, 16, 8, 4, 0.85),
    ],
)
def test_sdpa(device, num_tiles_k, num_tiles_v, chunk_size, num_chunks, scale):
    """Test TTNN SDPA Q*K^T operation on a single core"""

    # Tile dimensions
    tile_height = 8
    tile_width = 32
    q_tile = ttnn.Tile([tile_height, tile_width])
    k_tile = ttnn.Tile([32, 32])

    # Q shape: [1 tile height, num_tiles_k tiles width] = [32, num_tiles_k * 32]
    # K shape: [num_chunks * chunk_size tiles height, num_tiles_k tiles width]
    # Output shape: [1 tile height, num_chunks * chunk_size tiles width]
    q_shape = (tile_height, num_tiles_k * tile_width)
    k_shape = (num_chunks * chunk_size * tile_width, num_tiles_k * tile_width)
    out_shape = (tile_height, num_tiles_v * tile_width)
    stats_shape = (tile_height, tile_width)

    logger.info(f"Q shape: {q_shape}, K shape: {k_shape}, Output shape: {out_shape}")

    # Create input PyTorch tensors
    torch.manual_seed(42)

    # Q and K are query/key tensors
    torch_q = torch.randn(q_shape, dtype=torch.bfloat16) * 0.1
    torch_k = torch.randn(k_shape, dtype=torch.bfloat16) * 0.1

    # Compute reference output using PyTorch: Q @ K^T
    torch_out_expected, torch_max_expected, torch_sum_expected = SdpaSingleCore.golden(
        torch_q, torch_k, out_shape[1], scale=scale
    )

    logger.info(f"Expected output shape: {torch_out_expected.shape}")

    # Shard spec: single core
    q_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        q_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    q_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, q_shard_spec)

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

    stats_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        stats_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    stats_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, stats_shard_spec)

    # Create input tensors sharded on single core
    ttnn_q = ttnn.from_torch(
        torch_q,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=q_mem_config,
        tile=q_tile,
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
        tile=q_tile,
    )

    torch_stats = torch.zeros(stats_shape, dtype=torch.bfloat16)
    ttnn_stats = ttnn.from_torch(
        torch_stats,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=stats_mem_config,
        tile=q_tile,
    )

    logger.info("Created tensors sharded on single core")

    # Run SDPA operation
    logger.info("Running SDPA Q*K^T operation...")
    result = SdpaSingleCore.op(
        ttnn_q,
        ttnn_k,
        ttnn_out,
        ttnn_stats,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        num_tiles_k=num_tiles_k,
        num_tiles_v=num_tiles_v,
        scale=scale,
    )

    # Convert back to torch for verification
    out_torch, stats_torch = result
    out_torch = ttnn.to_torch(out_torch)
    stats_torch = ttnn.to_torch(stats_torch)
    max_torch = stats_torch[:, 0:1]
    sum_torch = stats_torch[:, 1:2]

    # Verify output shape
    assert out_torch.shape == out_shape, f"Expected output shape {out_shape}, got {out_torch.shape}"

    # Verify results
    logger.info("Verifying SDPA results...")
    # Check output
    out_max_diff = torch.max(torch.abs(out_torch - torch_out_expected)).item()
    out_mean_diff = torch.mean(torch.abs(out_torch - torch_out_expected)).item()
    max_max_diff = torch.max(torch.abs(max_torch - torch_max_expected)).item()
    max_mean_diff = torch.mean(torch.abs(max_torch - torch_max_expected)).item()
    sum_max_diff = torch.max(torch.abs(sum_torch - torch_sum_expected)).item()
    sum_mean_diff = torch.mean(torch.abs(sum_torch - torch_sum_expected)).item()
    logger.info(f"Out Max absolute difference: {out_max_diff}")
    logger.info(f"Out Mean absolute difference: {out_mean_diff}")
    logger.info(f"Max Max absolute difference: {max_max_diff}")
    logger.info(f"Max Mean absolute difference: {max_mean_diff}")
    logger.info(f"Sum Max absolute difference: {sum_max_diff}")
    logger.info(f"Sum Mean absolute difference: {sum_mean_diff}")

    passing, pcc_message = comp_pcc(torch_out_expected, out_torch, 0.99)
    logger.info(f"{pcc_message}")

    assert passing, f"Output PCC failed: {pcc_message}"

    passing, pcc_message = comp_pcc(torch_max_expected, max_torch, 0.99)
    logger.info(f"{pcc_message}")
    assert passing, f"Max PCC failed: {pcc_message}"

    passing, pcc_message = comp_pcc(torch_sum_expected, sum_torch, 0.99)
    logger.info(f"{pcc_message}")
    assert passing, f"Sum PCC failed: {pcc_message}"

    logger.info("✓ SDPA Q*K^T test passed!")
