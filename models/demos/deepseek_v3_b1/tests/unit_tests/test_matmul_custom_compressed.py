# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Custom Matmul with Compressed Weights - Single Core

Tiles are pre-sorted by format. The kernel reconfigures the unpacker
only once per format group instead of per tile.
"""

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.micro_ops.matmul_custom_compressed.op import MatmulCustomCompressed
from models.demos.deepseek_v3_b1.tests.unit_tests.test_eltwise_add_compressed import scale_tiles_for_mixed_formats


def _run_matmul_custom_compressed(
    device,
    M,
    K,
    N,
    formats,
    threshold=0.993,
    pcc_threshold=0.98,
):
    """Helper: run custom compressed A @ decompress(B_compressed) on single core."""
    torch.manual_seed(0)

    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N)).float()
    scale_tiles_for_mixed_formats(torch_b, formats)

    # Single core
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})

    # Compress B
    bfp0_mae = 1e-3 if "bfp0" in formats else 0.01
    assigner = CompressedTensorAssigner(metric="pcc", threshold=threshold, formats=formats, bfp0_mae_threshold=bfp0_mae)
    b_shard_spec = ttnn.ShardSpec(core_grid, [K, N], ttnn.ShardOrientation.ROW_MAJOR)
    b_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, b_shard_spec)
    ct = CompressedTensor.from_torch(torch_b, assigner, device=device, memory_config=b_mem_config)

    logger.info(f"Custom compressed B: {ct}")
    logger.info(f"Tile counts: {ct.tile_counts}")

    # Verify all requested formats are used
    counts = ct.tile_counts
    for fmt in formats:
        assert counts.get(fmt, 0) > 0, f"Expected tiles with format {fmt}, got counts: {counts}"

    # Golden: A @ B (original float)
    torch_expected = (torch_a.float() @ torch_b).bfloat16()

    # Input A
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

    # Output
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

    # Run custom compressed matmul
    ttnn_result = MatmulCustomCompressed.op(ttnn_a, ct, ttnn_output)

    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == (M, N), f"Expected shape ({M}, {N}), got {output_torch.shape}"

    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(pcc_message)
    assert passing, pcc_message


def test_matmul_custom_compressed_small(device):
    """[1, 64] x [64, 32], bfp8 only. K=2 tiles."""
    _run_matmul_custom_compressed(device, 1, 64, 32, formats=["bfp8"])


def test_matmul_custom_compressed_mixed(device):
    """[1, 256] x [256, 32], mixed bfp8+bfp4. K=8 tiles."""
    _run_matmul_custom_compressed(device, 1, 256, 32, formats=["bfp8", "bfp4"])


def test_matmul_custom_compressed_large(device):
    """[1, 7168] x [7168, 32], mixed bfp8+bfp4. DeepSeek shape."""
    _run_matmul_custom_compressed(device, 1, 7168, 32, formats=["bfp8", "bfp4"])


def test_matmul_custom_compressed_large_uniform(device):
    """[1, 7168] x [7168, 32], bfp8 only. DeepSeek shape."""
    _run_matmul_custom_compressed(device, 1, 7168, 32, formats=["bfp8"])


def test_matmul_custom_compressed_wide(device):
    """[1, 64] x [64, 64], bfp8. out_w=2."""
    _run_matmul_custom_compressed(device, 1, 64, 64, formats=["bfp8"])


def test_matmul_custom_compressed_wide_mixed(device):
    """[1, 256] x [256, 128], mixed bfp8+bfp4. out_w=4."""
    _run_matmul_custom_compressed(device, 1, 256, 128, formats=["bfp8", "bfp4"])
