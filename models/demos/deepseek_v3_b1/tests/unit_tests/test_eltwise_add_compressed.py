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


def scale_tiles_for_mixed_formats(b_torch, formats):
    """Adjust tiles so the assigner picks different formats.

    bfp8 tiles: high within-block dynamic range (elements span orders of magnitude)
    bfp4 tiles: moderate range (randn, shared exponent works ok for bfp4)
    bfp2 tiles: uniform values within each block (shared exponent covers all)
    """
    if len(formats) <= 1:
        return

    M, N = b_torch.shape
    num_fmts = len(formats)
    tiles_h, tiles_w = M // 32, N // 32

    for idx in range(tiles_h * tiles_w):
        tr, tc = idx // tiles_w, idx % tiles_w
        fmt = formats[idx % num_fmts]
        r0, r1 = tr * 32, (tr + 1) * 32
        c0, c1 = tc * 32, (tc + 1) * 32
        tile = b_torch[r0:r1, c0:c1]
        if fmt == "bfp8":
            # High within-block dynamic range: multiply each row by exponentially
            # increasing factors. Elements in the same 16-element block will span
            # several orders of magnitude, requiring bfp8's 7 mantissa bits.
            for r in range(32):
                tile[r, :] *= 2.0 ** (r % 16)
        elif fmt == "bfp2":
            # bfp2 = 1 mantissa bit + shared exponent per 16-element block.
            # Make each row a random power of 2 (shared exp) with random signs.
            # Values like ±4, ±0.25, ±16 etc — varied across rows but uniform within block.
            for r in range(32):
                exp = torch.randint(-3, 4, (1,)).float()
                signs = torch.sign(torch.randn(32))
                signs[signs == 0] = 1.0
                b_torch[r0 + r, c0:c1] = signs * (2.0**exp)
        elif fmt == "bfp0":
            # bfp0 = zero tile. Use tiny random values that round to zero
            # under the bfp0_mae_threshold (1e-3).
            b_torch[r0:r1, c0:c1] = torch.randn(32, 32) * 1e-3
        # bfp4: keep randn as-is


def _run_eltwise_add_compressed(
    device,
    M,
    N,
    formats,
    threshold=0.993,
    pcc_threshold=0.98,
    num_cores_h=1,
    num_cores_w=1,
    shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
):
    """Helper: run A (bf16) + decompress(B_compressed) = C (bf16)."""
    torch.manual_seed(42)

    a_torch = torch.randn(1, 1, M, N).bfloat16().float()
    b_torch = torch.randn(M, N).float()
    scale_tiles_for_mixed_formats(b_torch, formats)

    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_w - 1, num_cores_h - 1))]
    )

    bfp0_mae = 1e-3 if "bfp0" in formats else 0.01
    assigner = CompressedTensorAssigner(metric="pcc", threshold=threshold, formats=formats, bfp0_mae_threshold=bfp0_mae)
    b_mem_config = _make_sharded_mem_config((M, N), shard_layout, ttnn.BufferType.L1, core_grid)
    ct = CompressedTensor.from_torch(
        b_torch, assigner, device=device, memory_config=b_mem_config, assignment_memory_config=b_mem_config
    )

    logger.info(f"Compressed B: {ct}")
    logger.info(f"Tile counts: {ct.tile_counts}")

    # Verify the assigner actually used all requested formats
    counts = ct.tile_counts
    for fmt in formats:
        assert counts.get(fmt, 0) > 0, f"Expected tiles with format {fmt}, got counts: {counts}"

    # Golden: A + B (original float, no quantization)
    golden = a_torch + b_torch.unsqueeze(0).unsqueeze(0)

    # A tensor on device
    a_mem_config = _make_sharded_mem_config((M, N), shard_layout, ttnn.BufferType.L1, core_grid)
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


# --- Single-core tests ---


def test_eltwise_add_compressed_1tile_bfp2(device):
    """1 tile, bfp2 only."""
    _run_eltwise_add_compressed(device, 32, 32, formats=["bfp2"], pcc_threshold=0.94)


def test_eltwise_add_compressed_4tile_bfp2(device):
    """4 tiles (64x64), bfp2 only."""
    _run_eltwise_add_compressed(device, 64, 64, formats=["bfp2"], pcc_threshold=0.94)


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


def test_eltwise_add_compressed_16tile_mixed_all_formats(device):
    """16 tiles (128x128), mixed bfp2 + bfp8."""
    _run_eltwise_add_compressed(
        device, 128, 128, formats=["bfp8", "bfp4", "bfp2"], threshold=0.994, pcc_threshold=0.997
    )


# --- Multi-core HEIGHT_SHARDED tests ---


def test_eltwise_add_compressed_2core_bfp8(device):
    """2 cores, 64x32 (1 tile/core), bfp8."""
    _run_eltwise_add_compressed(device, 64, 32, formats=["bfp8"], num_cores_h=2)


def test_eltwise_add_compressed_4core_mixed(device):
    """4 cores, 128x32 (1 tile/core), mixed bfp4 + bfp8."""
    _run_eltwise_add_compressed(device, 128, 32, formats=["bfp8", "bfp4"], num_cores_h=4)


def test_eltwise_add_compressed_2core_4tile_mixed(device):
    """2 cores, 128x64 (4 tiles/core), mixed bfp4 + bfp8."""
    _run_eltwise_add_compressed(device, 128, 64, formats=["bfp8", "bfp4"], num_cores_h=2)


def test_eltwise_add_compressed_8core_all_formats(device):
    """8 cores, 256x128 (4 tiles/core), all formats."""
    _run_eltwise_add_compressed(
        device, 256, 128, formats=["bfp8", "bfp4", "bfp2"], num_cores_h=8, threshold=0.994, pcc_threshold=0.997
    )


# --- Uneven sharding tests ---


def test_eltwise_add_compressed_3core_uneven(device):
    """3 cores, 96x32. 96/3 = 32 rows/core, evenly split."""
    _run_eltwise_add_compressed(device, 96, 32, formats=["bfp8", "bfp4"], num_cores_h=3)


def test_eltwise_add_compressed_3core_uneven_large(device):
    """128/3 -> div_up=43 -> align to 64. Last core gets padded."""
    _run_eltwise_add_compressed(
        device, 128, 64, formats=["bfp8", "bfp4", "bfp2"], num_cores_h=3, threshold=0.994, pcc_threshold=0.997
    )


def test_eltwise_add_compressed_5core_uneven(device):
    """192/5 -> div_up=39 -> align to 64. Last cores get padded."""
    _run_eltwise_add_compressed(
        device, 192, 32, formats=["bfp8", "bfp4", "bfp2"], num_cores_h=5, threshold=0.994, pcc_threshold=0.997
    )


# --- WIDTH_SHARDED tests ---


def test_eltwise_add_compressed_2core_width(device):
    """2 cores width-sharded, 32x64 (1 tile/core), bfp8."""
    _run_eltwise_add_compressed(
        device,
        32,
        64,
        formats=["bfp8"],
        num_cores_w=2,
        shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    )


def test_eltwise_add_compressed_4core_width_mixed(device):
    """4 cores width-sharded, 64x128 (2 tiles/core), mixed bfp4 + bfp8."""
    _run_eltwise_add_compressed(
        device,
        64,
        128,
        formats=["bfp8", "bfp4"],
        num_cores_w=4,
        shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    )


def test_eltwise_add_compressed_3core_width_uneven(device):
    """3 cores width-sharded, 64x96, uneven. div_up(96,3)=32."""
    _run_eltwise_add_compressed(
        device,
        64,
        96,
        formats=["bfp8", "bfp4"],
        num_cores_w=3,
        shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    )


def test_eltwise_add_compressed_4core_width_all_formats(device):
    """4 cores width-sharded, 64x128, all formats."""
    _run_eltwise_add_compressed(
        device,
        64,
        128,
        formats=["bfp8", "bfp4", "bfp2"],
        num_cores_w=4,
        shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        threshold=0.994,
        pcc_threshold=0.997,
    )


# --- BLOCK_SHARDED tests ---


def test_eltwise_add_compressed_2x2_block(device):
    """2x2 block-sharded, 64x64 (1 tile/core), bfp8."""
    _run_eltwise_add_compressed(
        device,
        64,
        64,
        formats=["bfp8"],
        num_cores_h=2,
        num_cores_w=2,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )


def test_eltwise_add_compressed_2x2_block_mixed(device):
    """2x2 block-sharded, 128x128 (4 tiles/core), mixed bfp4 + bfp8."""
    _run_eltwise_add_compressed(
        device,
        128,
        128,
        formats=["bfp8", "bfp4"],
        num_cores_h=2,
        num_cores_w=2,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )


def test_eltwise_add_compressed_2x2_block_all_formats(device):
    """2x2 block-sharded, 128x128 (4 tiles/core), all formats."""
    _run_eltwise_add_compressed(
        device,
        128,
        128,
        formats=["bfp8", "bfp4", "bfp2"],
        num_cores_h=2,
        num_cores_w=2,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        threshold=0.994,
        pcc_threshold=0.997,
    )


def test_eltwise_add_compressed_2x3_block_uneven(device):
    """2x3 block-sharded, 64x128, uneven width. div_up(128,3)=43 -> align 64."""
    _run_eltwise_add_compressed(
        device,
        64,
        128,
        formats=["bfp8", "bfp4", "bfp2"],
        num_cores_h=2,
        num_cores_w=3,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        threshold=0.994,
        pcc_threshold=0.997,
    )


def test_eltwise_add_compressed_3x2_block_uneven(device):
    """3x2 block-sharded, 128x64, uneven height. div_up(128,3)=43 -> align 64."""
    _run_eltwise_add_compressed(
        device,
        128,
        64,
        formats=["bfp8", "bfp4", "bfp2"],
        num_cores_h=3,
        num_cores_w=2,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        threshold=0.994,
        pcc_threshold=0.997,
    )


# --- BFP0 tests (zero tiles) ---


def test_eltwise_add_compressed_4tile_mixed_bfp80(device):
    """4 tiles (64x64), mixed bfp8 + bfp0."""
    _run_eltwise_add_compressed(device, 64, 64, formats=["bfp8", "bfp0"])


def test_eltwise_add_compressed_2core_mixed_bfp80(device):
    """2 cores height-sharded, 64x64 (2 tiles/core), mixed bfp8 + bfp0."""
    _run_eltwise_add_compressed(device, 64, 64, formats=["bfp8", "bfp0"], num_cores_h=2)


def test_eltwise_add_compressed_16tile_mixed_bfp8420(device):
    """4 tiles (128x128), all formats bfp8 + bfp4 + bfp2 + bfp0."""
    _run_eltwise_add_compressed(
        device, 128, 128, formats=["bfp8", "bfp4", "bfp2", "bfp0"], threshold=0.994, pcc_threshold=0.997
    )


def test_eltwise_add_compressed_2x2_block_all_with_bfp0(device):
    """2x2 block-sharded, 128x128, all formats bfp8 + bfp4 + bfp2 + bfp0."""
    _run_eltwise_add_compressed(
        device,
        128,
        128,
        formats=["bfp8", "bfp4", "bfp2", "bfp0"],
        num_cores_h=2,
        num_cores_w=2,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        threshold=0.994,
        pcc_threshold=0.997,
    )
