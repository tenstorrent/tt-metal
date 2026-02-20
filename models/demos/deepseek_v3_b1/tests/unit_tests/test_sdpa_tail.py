# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN sdpa_tail Test

Tests the sdpa_tail function which computes:
    m = max(m1, m2)
    P1 = exp((m1 - m) * scale)
    P2 = exp((m2 - m) * scale)
    s = s1 * P1 + s2 * P2
    l = l1 * P1 + l2 * P2

Input tensors are sharded on a single core.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.sdpa_tail.op import SdpaTailSingleCore


@pytest.mark.parametrize(
    "width, block_size, num_blocks",
    [
        (16, 8, 2),  # Multiple columns
    ],
)
@pytest.mark.parametrize("final_reduction", [True, False])
@pytest.mark.parametrize("scale", [192**-0.5])
def test_sdpa_tail(device, width, block_size, num_blocks, final_reduction, scale):
    """Test TTNN sdpa_tail operation on a single core"""

    # Tile dimensions
    tile_height = 8
    tile_width = 32
    tile = ttnn.Tile([tile_height, tile_width])
    height = 1

    assert width // num_blocks == block_size, f"Width {width} must be divisible by num_blocks {num_blocks}"

    # Shape for l tensors: [height * tile_height, width * tile_width]
    l_shape = (height * tile_height, width * tile_width)
    # Shape for m and s tensors: [height * tile_height, tile_width]
    # Values in column 0 are broadcast across columns when multiplying with l tensors
    ms_shape = (height * tile_height, tile_width)

    logger.info(f"L tensor shape: {l_shape}, M/S tensor shape: {ms_shape}")

    # Create input PyTorch tensors with reasonable values
    torch.manual_seed(42)

    # l1 and l2 are accumulator values (can be any reasonable values)
    torch_l1 = torch.randn(l_shape, dtype=torch.bfloat16) * 0.1
    torch_l2 = torch.randn(l_shape, dtype=torch.bfloat16) * 0.1

    # m1 and m2 are max values - use small negative values typical for softmax
    # The meaningful values are in column 0 of each row
    torch_ms1 = torch.zeros(ms_shape, dtype=torch.bfloat16)
    torch_ms2 = torch.zeros(ms_shape, dtype=torch.bfloat16)
    torch_ms1[:, 0] = torch.randn(ms_shape[0], dtype=torch.bfloat16) * 0.5 - 1.0
    torch_ms2[:, 0] = torch.randn(ms_shape[0], dtype=torch.bfloat16) * 0.5 - 1.0

    # s1 and s2 are sum values (positive)
    torch_ms1[:, 1] = torch.abs(torch.randn(ms_shape[0], dtype=torch.bfloat16)) + 0.1
    torch_ms2[:, 1] = torch.abs(torch.randn(ms_shape[0], dtype=torch.bfloat16)) + 0.1

    # Compute reference output using PyTorch
    # For golden, we need to broadcast m/s across l dimensions
    m1_bcast = torch_ms1[:, 0:1]
    m2_bcast = torch_ms2[:, 0:1]
    s1_bcast = torch_ms1[:, 1:2]
    s2_bcast = torch_ms2[:, 1:2]

    torch_l_expected, torch_m_expected, torch_s_expected = SdpaTailSingleCore.golden(
        torch_l1, torch_l2, m1_bcast, m2_bcast, s1_bcast, s2_bcast, scale=scale, final_reduction=final_reduction
    )

    # Shard spec: single core
    l_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        l_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    l_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l_shard_spec)

    ms_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ms_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    ms_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ms_shard_spec)

    # Create input tensors sharded on single core
    ttnn_l1 = ttnn.from_torch(
        torch_l1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=l_mem_config,
        tile=tile,
    )
    ttnn_l2 = ttnn.from_torch(
        torch_l2,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=l_mem_config,
        tile=tile,
    )
    ttnn_ms1 = ttnn.from_torch(
        torch_ms1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ms_mem_config,
        tile=tile,
    )
    ttnn_ms2 = ttnn.from_torch(
        torch_ms2,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ms_mem_config,
        tile=tile,
    )

    # Create output tensors sharded on same core
    torch_l_out = torch.zeros(l_shape, dtype=torch.bfloat16)
    ttnn_l_out = ttnn.from_torch(
        torch_l_out,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=l_mem_config,
        tile=tile,
    )
    torch_ms_out = torch.zeros(ms_shape, dtype=torch.bfloat16)
    ttnn_ms_out = ttnn.from_torch(
        torch_ms_out,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ms_mem_config,
        tile=tile,
    )

    logger.info("Created tensors sharded on single core")

    # Run sdpa_tail operation
    logger.info("Running sdpa_tail operation...")
    l_result, ms_result = SdpaTailSingleCore.op(
        ttnn_l1,
        ttnn_l2,
        ttnn_ms1,
        ttnn_ms2,
        ttnn_l_out,
        ttnn_ms_out,
        scale=scale,
        block_size=block_size,
        num_blocks=num_blocks,
        final_reduction=final_reduction,
    )

    # Convert back to torch for verification
    l_out_torch = ttnn.to_torch(l_result)
    ms_out_torch = ttnn.to_torch(ms_result)

    # Verify output shapes
    assert l_out_torch.shape == l_shape, f"Expected L shape {l_shape}, got {l_out_torch.shape}"
    assert ms_out_torch.shape == ms_shape, f"Expected M/S shape {ms_shape}, got {ms_out_torch.shape}"

    # Verify sdpa_tail results
    logger.info("Verifying sdpa_tail results...")

    m_out_torch = ms_out_torch[:, 0:1]
    s_out_torch = ms_out_torch[:, 1:2]

    # Check L output
    max_diff_l = torch.max(torch.abs(l_out_torch - torch_l_expected)).item()
    mean_diff_l = torch.mean(torch.abs(l_out_torch - torch_l_expected)).item()
    logger.info(f"L - Max absolute difference: {max_diff_l}")
    logger.info(f"L - Mean absolute difference: {mean_diff_l}")

    passing_l, pcc_message_l = comp_pcc(torch_l_expected, l_out_torch, 0.999)
    logger.info(f"L - {pcc_message_l}")

    if not final_reduction:
        # Check M output (only column 0 matters)
        max_diff_m = torch.max(torch.abs(m_out_torch - torch_m_expected)).item()
        logger.info(f"M - Max absolute difference (col 0): {max_diff_m}")

        passing_m, pcc_message_m = comp_pcc(torch_m_expected, m_out_torch, 0.999)
        logger.info(f"M - {pcc_message_m}")

        # Check S output (only column 0 matters)
        max_diff_s = torch.max(torch.abs(s_out_torch - torch_s_expected)).item()
        logger.info(f"S - Max absolute difference (col 0): {max_diff_s}")

        passing_s, pcc_message_s = comp_pcc(torch_s_expected, s_out_torch, 0.999)
        logger.info(f"S - {pcc_message_s}")

    assert passing_l, f"L output PCC failed: {pcc_message_l}"
    if not final_reduction:
        assert passing_m, f"M output PCC failed: {pcc_message_m}"
        assert passing_s, f"S output PCC failed: {pcc_message_s}"

    logger.info("✓ sdpa_tail test passed!")
