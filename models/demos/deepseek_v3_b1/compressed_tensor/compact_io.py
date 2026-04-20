# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compact on-disk format for BSPM CompressedTensor tile data.

Stores tiles in logical (pre-DRAM-shuffle) row-major order with variable
byte lengths per tile — BFP4=576 B, BFP2=320 B, zero=0 B — yielding disk
representations that are measurably smaller than uniform bfloat4_b.

Disk space comparison for a gate_proj expert (K=7168, N=2048, 14 336 tiles):
    Uniform BFP4 (DRAM-padded)  : ~7.9 MB
    BSPM 3.5 b/e compact        : ~4.8 MB  (60 % BFP4 + 25 % BFP2 + 15 % zero)

Across all 256 experts × 59 MoE layers × 3 projections for R1-0528, compact
storage saves ~90 GB vs. either a uniform BFP4 cache or the legacy padded cache.
"""

from __future__ import annotations

import io

import numpy as np

from .tile_utils import (
    BFP_MANT_BITS,
    COMPRESSED_FORMATS,
    DEFAULT_TILE_HW,
    bfp_tile_packed_size,
    pack_bfp_tile,
    unpack_bfp_tile,
)

# Format index → mantissa bits  (0=bfp8/7, 1=bfp4/3, 2=bfp2/1, 3=bfp0/0)
_FMT_TO_MANT: dict[int, int] = {
    idx: BFP_MANT_BITS[fmt] for idx, fmt in enumerate(COMPRESSED_FORMATS) if fmt in BFP_MANT_BITS
}


def pack_compact_tiles(
    w_kn: np.ndarray,
    assignment_2d: np.ndarray,
    tile_hw: int = DEFAULT_TILE_HW,
) -> bytes:
    """Pack a (K, N) weight matrix to compact tile bytes in logical row-major order.

    Zero tiles (code 3 / bfp0) contribute 0 bytes — they are skipped entirely.
    The returned byte string is tightly packed; use *assignment_2d* with
    :func:`unpack_compact_tiles` to reconstruct tile offsets during load.

    Args:
        w_kn: ``(K, N)`` float32 numpy array in logical tile order (pre-DRAM-shuffle).
        assignment_2d: ``(tiles_h, tiles_w)`` int8 tile format codes.
        tile_hw: Tile dimension (default 32).

    Returns:
        Flat ``bytes``, variable-length per tile, logical row-major order.
    """
    if tile_hw != DEFAULT_TILE_HW:
        raise ValueError(f"tile_hw={tile_hw} is not supported; underlying BFP primitives require {DEFAULT_TILE_HW}")
    K, N = w_kn.shape
    if K % tile_hw != 0 or N % tile_hw != 0:
        raise ValueError(
            f"w_kn shape {w_kn.shape} must be divisible by tile_hw={tile_hw}; "
            f"got remainders K%tile_hw={K % tile_hw}, N%tile_hw={N % tile_hw}"
        )
    tiles_h = K // tile_hw
    tiles_w = N // tile_hw
    if assignment_2d.shape != (tiles_h, tiles_w):
        raise ValueError(f"assignment_2d {assignment_2d.shape} does not match tile grid ({tiles_h}, {tiles_w})")
    buf = io.BytesIO()
    for r in range(tiles_h):
        for c in range(tiles_w):
            mant_bits = _FMT_TO_MANT.get(int(assignment_2d[r, c]), 0)
            if mant_bits == 0:
                continue  # zero tile — no bytes written
            tile = w_kn[r * tile_hw : (r + 1) * tile_hw, c * tile_hw : (c + 1) * tile_hw]
            tile_bytes = pack_bfp_tile(tile.astype(np.float32), mant_bits)
            buf.write(tile_bytes.tobytes())
    return buf.getvalue()


def unpack_compact_tiles(
    stream: bytes,
    assignment_2d: np.ndarray,
    tile_hw: int = DEFAULT_TILE_HW,
) -> np.ndarray:
    """Reconstruct a (K, N) float32 weight matrix from compact tile bytes.

    Zero tiles are filled with ``0.0``.  *stream* must have been produced by
    :func:`pack_compact_tiles` with the same *assignment_2d*.

    Args:
        stream: Compact byte string from :func:`pack_compact_tiles`.
        assignment_2d: ``(tiles_h, tiles_w)`` int8 tile format codes.
        tile_hw: Tile dimension (default 32).

    Returns:
        ``(K, N)`` float32 numpy array in logical tile order.
    """
    if tile_hw != DEFAULT_TILE_HW:
        raise ValueError(f"tile_hw={tile_hw} is not supported; underlying BFP primitives require {DEFAULT_TILE_HW}")
    tiles_h, tiles_w = int(assignment_2d.shape[0]), int(assignment_2d.shape[1])
    expected_bytes = compact_tile_byte_count(assignment_2d, tile_hw)
    if len(stream) != expected_bytes:
        raise ValueError(
            f"stream length {len(stream)} does not match assignment-implied size {expected_bytes}; "
            "stream may be truncated or corrupt"
        )
    K = tiles_h * tile_hw
    N = tiles_w * tile_hw
    w_out = np.zeros((K, N), dtype=np.float32)
    buf = memoryview(stream)
    offset = 0
    for r in range(tiles_h):
        for c in range(tiles_w):
            mant_bits = _FMT_TO_MANT.get(int(assignment_2d[r, c]), 0)
            if mant_bits == 0:
                continue  # zero tile — already 0.0
            nbytes = bfp_tile_packed_size(mant_bits, tile_hw)
            tile_bytes = np.frombuffer(buf[offset : offset + nbytes], dtype=np.uint8).copy()
            tile = unpack_bfp_tile(tile_bytes, mant_bits)
            w_out[r * tile_hw : (r + 1) * tile_hw, c * tile_hw : (c + 1) * tile_hw] = tile
            offset += nbytes
    return w_out


def compact_tile_byte_count(
    assignment_2d: np.ndarray,
    tile_hw: int = DEFAULT_TILE_HW,
) -> int:
    """Return the number of bytes that :func:`pack_compact_tiles` would produce.

    Used for size assertions (e.g. assert compact < bfp4_baseline in tests).
    Does not require the weight tensor — only the assignment is needed.
    """
    total = 0
    for code in assignment_2d.ravel():
        mant_bits = _FMT_TO_MANT.get(int(code), 0)
        total += bfp_tile_packed_size(mant_bits, tile_hw)
    return total


def bfp4_tile_byte_count(tiles_h: int, tiles_w: int, tile_hw: int = DEFAULT_TILE_HW) -> int:
    """Return the byte count for a fully uniform BFP4 tensor with the given tile grid.

    Useful as the baseline for compression ratio assertions.
    """
    return tiles_h * tiles_w * bfp_tile_packed_size(BFP_MANT_BITS["bfp4"], tile_hw)
