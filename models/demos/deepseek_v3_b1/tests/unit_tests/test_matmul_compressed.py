# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SRAM Matmul with Compressed Weights - Single Core

Tests matmul where input B (weights) is a CompressedTensor with mixed BFP formats.
Computes: output = A @ decompress(B_compressed)

- Input A (in0): bfloat16 HEIGHT_SHARDED, [M, K] per core
- Input B (in1): CompressedTensor (mixed bfp8/bfp4/bfp2/bfp0), WIDTH_SHARDED, [K, N] per core
- Output: bfloat16 WIDTH_SHARDED, [M, N] per core
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.micro_ops.matmul_compressed.op import MatmulCompressed
from models.demos.deepseek_v3_b1.tests.unit_tests.test_eltwise_add_compressed import scale_tiles_for_mixed_formats


def _run_matmul_compressed(
    device,
    M,
    K,
    N,
    formats,
    threshold=0.993,
    pcc_threshold=0.98,
):
    """Helper: run A @ decompress(B_compressed) on single core."""
    torch.manual_seed(0)

    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N)).float()
    scale_tiles_for_mixed_formats(torch_b, formats)

    # Single core
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})

    # Compress B (weights)
    assigner = CompressedTensorAssigner(metric="pcc", threshold=threshold, formats=formats)
    b_shard_spec = ttnn.ShardSpec(core_grid, [K, N], ttnn.ShardOrientation.ROW_MAJOR)
    b_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, b_shard_spec)
    ct = CompressedTensor.from_torch(torch_b, assigner, device=device, memory_config=b_mem_config)

    logger.info(f"Compressed B: {ct}")
    logger.info(f"Tile counts: {ct.tile_counts}")

    # Verify all requested formats are used
    counts = ct.tile_counts
    for fmt in formats:
        assert counts.get(fmt, 0) > 0, f"Expected tiles with format {fmt}, got counts: {counts}"

    # Golden: A @ B (original float)
    torch_expected = (torch_a.float() @ torch_b).bfloat16()

    # Input A: HEIGHT_SHARDED on single core
    a_tile = ttnn.Tile([M, 32])
    a_shard_spec = ttnn.ShardSpec(core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR)
    a_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec)
    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=a_mem_config,
        tile=a_tile,
    )

    # Output: WIDTH_SHARDED on single core
    out_tile = ttnn.Tile([M, 32])
    out_shard_spec = ttnn.ShardSpec(core_grid, [M, N], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec)
    ttnn_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem_config,
        tile=out_tile,
    )

    logger.info(f"A shape: [{M}, {K}], B compressed shape: [{K}, {N}], Output shape: [{M}, {N}]")

    # Run matmul compressed
    ttnn_result = MatmulCompressed.op(ttnn_a, ct, ttnn_output)

    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == (M, N), f"Expected shape ({M}, {N}), got {output_torch.shape}"

    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(pcc_message)
    assert passing, pcc_message


# --- Small debug test ---


def test_matmul_compressed_small(device):
    """Small matmul: [1, 64] x [64, 32], bfp8 only."""
    _run_matmul_compressed(device, 1, 64, 32, formats=["bfp8"])


# --- Single-core tests (DeepSeek v3 shapes) ---


@pytest.mark.parametrize(
    "M, K, N, formats",
    [
        # MoE gate/up proj: [1, 7168] x [7168, 32]
        (1, 7168, 32, ["bfp8", "bfp4"]),
        # MoE down proj: [1, 2048] x [2048, 32]
        (1, 2048, 32, ["bfp8", "bfp4"]),
        # Q down + K down: [1, 7168] x [7168, 32]
        (1, 7168, 32, ["bfp8"]),
        # Out proj: [1, 8192] x [8192, 64]
        (1, 8192, 64, ["bfp8", "bfp4"]),
        # LM Head: [1, 7168] x [7168, 128]
        (1, 7168, 128, ["bfp8", "bfp4"]),
    ],
)
def test_matmul_compressed_single_core(device, M, K, N, formats):
    """Single-core matmul with compressed weights."""
    _run_matmul_compressed(device, M, K, N, formats)


@pytest.mark.parametrize(
    "M, K, N, formats",
    [
        # Small shape, all formats
        (1, 7168, 32, ["bfp8", "bfp4", "bfp2", "bfp0"]),
        # Medium shape with bfp0
        (1, 8192, 64, ["bfp8", "bfp4", "bfp2", "bfp0"]),
    ],
)
def test_matmul_compressed_all_formats(device, M, K, N, formats):
    """Single-core matmul with all compressed formats."""
    _run_matmul_compressed(device, M, K, N, formats, threshold=0.994, pcc_threshold=0.97)
