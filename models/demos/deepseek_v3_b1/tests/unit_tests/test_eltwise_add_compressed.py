# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test eltwise add with compressed tensor input.

Single core, HEIGHT_SHARDED.
"""

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.micro_ops.eltwise_add_compressed.op import EltwiseAddCompressed
from models.demos.deepseek_v3_b1.tests.unit_tests.test_compressed_tensor import _make_sharded_mem_config
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def _run_eltwise_add_compressed(device, M, N, formats, threshold=0.993, pcc_threshold=0.98):
    """Helper: run A (bf16) + decompress(B_compressed) = C (bf16) on single core."""
    torch.manual_seed(42)

    a_torch = torch.randn(1, 1, M, N).bfloat16().float()
    b_torch = torch.randn(M, N).float()
    # Scale alternating tiles so the assigner picks different formats:
    # even tiles get large range (needs bfp8), odd tiles get small range (bfp4 ok)
    tiles_h = M // 32
    tiles_w = N // 32
    for tr in range(tiles_h):
        for tc in range(tiles_w):
            if (tr + tc) % 2 == 1:
                b_torch[tr * 32 : (tr + 1) * 32, tc * 32 : (tc + 1) * 32] *= 0.5

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    assigner = CompressedTensorAssigner(metric="pcc", threshold=threshold, formats=formats)
    b_mem_config = _make_sharded_mem_config(
        (M, N), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, core_grid
    )
    ct = CompressedTensor.from_torch(b_torch, assigner, device=device, memory_config=b_mem_config)

    logger.info(f"Compressed B: {ct}")
    logger.info(f"Tile counts: {ct.tile_counts}")

    # Golden: A + dequantized(B)
    b_decompressed = ct.to_torch()
    golden = a_torch + b_decompressed.unsqueeze(0).unsqueeze(0)

    # A tensor on device
    a_shard_spec = ttnn.ShardSpec(core_grid, [M, N], ttnn.ShardOrientation.ROW_MAJOR)
    a_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard_spec)
    a_t = ttnn.from_torch(
        a_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=a_mem_config
    )

    # Output tensor on device
    out_t = ttnn.from_torch(
        torch.zeros_like(a_torch),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=a_mem_config,
    )

    # Run
    result_t = EltwiseAddCompressed.op(a_t, ct, out_t)

    # Compare
    result_torch = ttnn.to_torch(result_t)
    passing, output = comp_pcc(golden, result_torch, pcc_threshold)

    logger.info(output)
    assert passing, f"PCC check failed: {output}"


def test_eltwise_add_compressed_1tile_bfp4(device):
    """1 tile, bfp4 only."""
    _run_eltwise_add_compressed(device, 32, 32, formats=["bfp4"])


def test_eltwise_add_compressed_4tile_bfp4(device):
    """4 tiles (64x64), bfp4 only."""
    _run_eltwise_add_compressed(device, 64, 64, formats=["bfp4"])


def test_eltwise_add_compressed_1tile_bfp8(device):
    """1 tile, bfp8 only."""
    _run_eltwise_add_compressed(device, 32, 32, formats=["bfp8"])


def test_eltwise_add_compressed_4tile_bfp8(device):
    """4 tiles (64x64), bfp8 only."""
    _run_eltwise_add_compressed(device, 64, 64, formats=["bfp8"])


def test_eltwise_add_compressed_2tile_mixed(device):
    """2 tiles (64x32), mixed bfp4 + bfp8."""
    _run_eltwise_add_compressed(device, 64, 32, formats=["bfp8", "bfp4"])


def test_eltwise_add_compressed_4tile_mixed(device):
    """4 tiles (64x64), mixed bfp4 + bfp8."""
    _run_eltwise_add_compressed(device, 64, 64, formats=["bfp8", "bfp4"])


def test_eltwise_add_compressed_16tile_mixed(device):
    """16 tiles (128x128), mixed bfp4 + bfp8."""
    _run_eltwise_add_compressed(device, 128, 128, formats=["bfp8", "bfp4"])
