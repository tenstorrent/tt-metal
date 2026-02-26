# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for storing mixed-precision tile tensors through ttnn.

Flow:
  1. Start with a float32 torch tensor.
  2. Run mixed-tile assignment (bfp8/bfp4) to quantize each tile.
  3. Pack tiles into raw BFP bytes via C++ pack functions.
  4. Concatenate into a flat uint8 buffer, store as ttnn tensor.
  5. Read back, unpack each tile via C++ unpack functions.
  6. Validate PCC against original.
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.mixed_tile.assigner import MixedTileAssigner
from models.demos.deepseek_v3_b1.mixed_tile.metrics import metric_value
from models.demos.deepseek_v3_b1.mixed_tile.tile_utils import (
    MIXED_TILE_FORMATS,
    pack_mixed_tiles,
    ttnn_quantize_fn,
    unpack_mixed_tiles,
)


def test_mixed_tile_pack_store_unpack():
    """Pack tiles with mixed bfp4/bfp8, store as flat uint8 in ttnn, unpack and validate."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)

    # Threshold 0.995 is above bfp4 PCC (~0.993), so some tiles need bfp8
    assigner = MixedTileAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    result = assigner.assign(x, ttnn_quantize_fn)
    logger.info(f"Tile counts: {result.tile_counts}")
    logger.info(f"Assignment:\n{result.assignment}")
    assert (
        result.tile_counts["bfp8"] > 0 and result.tile_counts["bfp4"] > 0
    ), f"Expected a mix of bfp8 and bfp4: {result.tile_counts}"

    tile_hw = 32
    tiles_h = x.shape[0] // tile_hw
    tiles_w = x.shape[1] // tile_hw

    # Pack all tiles into flat uint8 torch tensor
    flat_buffer, tile_mant_bits = pack_mixed_tiles(x, result.assignment)
    logger.info(f"Total packed size: {flat_buffer.numel()} bytes")

    # Store as ttnn uint8 tensor and read back
    tt_tensor = ttnn.from_torch(flat_buffer.unsqueeze(0), dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT)
    recovered_flat = ttnn.to_torch(tt_tensor).squeeze()
    assert torch.equal(flat_buffer, recovered_flat), "Flat buffer round-trip mismatch"
    logger.info("Flat uint8 buffer: ttnn round-trip is bit-exact")

    # Unpack back to float32 torch tensor
    reconstructed = unpack_mixed_tiles(recovered_flat, tile_mant_bits, tiles_h, tiles_w)

    # Validate overall PCC
    overall_pcc = metric_value(x.numpy(), reconstructed.numpy(), "pcc")
    logger.info(f"Overall PCC: {overall_pcc:.6f}")
    assert overall_pcc > 0.98, f"Overall PCC {overall_pcc:.6f} unexpectedly low"

    # Log per-tile PCC
    for tr in range(tiles_h):
        for tc in range(tiles_w):
            ref_tile = x[tr * tile_hw : (tr + 1) * tile_hw, tc * tile_hw : (tc + 1) * tile_hw]
            rec_tile = reconstructed[tr * tile_hw : (tr + 1) * tile_hw, tc * tile_hw : (tc + 1) * tile_hw]
            tile_pcc = metric_value(ref_tile.numpy(), rec_tile.numpy(), "pcc")
            fmt_name = MIXED_TILE_FORMATS[result.assignment[tr, tc]]
            logger.info(f"  tile ({tr},{tc}) [{fmt_name}]: PCC={tile_pcc:.6f}")
