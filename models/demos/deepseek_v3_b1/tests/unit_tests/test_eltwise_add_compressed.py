# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test eltwise add with compressed tensor input.

A (bf16 TILE_LAYOUT) + B (compressed bfp8/bfp4 mix) = C (bf16 TILE_LAYOUT)

Single core, HEIGHT_SHARDED, 128x128 tensor (4x4 tile grid).
"""

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.micro_ops.eltwise_add_compressed.op import EltwiseAddCompressed
from models.demos.deepseek_v3_b1.tests.unit_tests.test_compressed_tensor import _make_sharded_mem_config
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def test_eltwise_add_compressed_single_core(device):
    """
    Single-core eltwise add: A (bf16) + decompress(B_compressed) = C (bf16).
    """
    torch.manual_seed(42)
    M, N = 128, 128  # 4x4 tile grid

    # Create input tensors
    a_torch = torch.randn(1, 1, M, N).bfloat16().float()
    b_torch = torch.randn(M, N).float()

    # Single core grid
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    # Create compressed tensor B, HEIGHT_SHARDED on single core in L1
    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8"])
    b_mem_config = _make_sharded_mem_config(
        (M, N), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, core_grid
    )
    ct = CompressedTensor.from_torch(b_torch, assigner, device=device, memory_config=b_mem_config)

    logger.info(f"Compressed B: {ct}")
    logger.info(f"Tile counts: {ct.tile_counts}")

    # Only bfp8/bfp4 supported in the kernel for now (no bfp2/bfp0 unpack on device yet)
    counts = ct.tile_counts
    assert counts.get("bfp2", 0) == 0, f"bfp2 tiles not supported in kernel: {counts}"
    assert counts.get("bfp0", 0) == 0, f"bfp0 tiles not supported in kernel: {counts}"
    assert counts["bfp8"] > 0 or counts["bfp4"] > 0, f"Expected bfp8/bfp4 tiles: {counts}"

    # Golden: A + dequantized(B)
    b_decompressed = ct.to_torch()
    golden = a_torch + b_decompressed.unsqueeze(0).unsqueeze(0)

    # A tensor on device (HEIGHT_SHARDED, single core)
    a_shard_spec = ttnn.ShardSpec(core_grid, [M, N], ttnn.ShardOrientation.ROW_MAJOR)
    a_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec)
    a_t = ttnn.from_torch(
        a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=a_mem_config
    )

    # Output tensor on device
    out_t = ttnn.from_torch(
        torch.zeros_like(a_torch),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=a_mem_config,
    )

    logger.info(f"A L1 addr: {a_t.buffer_address():#x}")
    logger.info(f"B data L1 addr: {ct.get_data_l1_address():#x}")
    logger.info(f"B assign L1 addr: {ct.get_assignment_l1_address():#x}")
    logger.info(f"Out L1 addr: {out_t.buffer_address():#x}")

    # Run eltwise add compressed
    result_t = EltwiseAddCompressed.op(a_t, ct, out_t)

    # Compare
    result_torch = ttnn.to_torch(result_t)
    passing, output = comp_pcc(golden, result_torch, 0.98)

    logger.info(output)
    logger.info(golden)
    logger.info(result_torch)
    assert passing, f"PCC check failed: {output}"
