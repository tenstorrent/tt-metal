# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ternary weights as Bfp2_b tile data with no exponent section (the unpacker forces one shared exponent).

A Bfp2_b element is 2 bits, sign in the high bit and a 1-bit mantissa in the low one (tt-metal
tt_metal/impl/data_format/bfloat2.cpp). Ternary t in {-1, 0, +1} is code 0b11, 0b00, 0b01. A 32x32 tile is four
16x16 faces (top-left, top-right, bottom-left, bottom-right), each face row one uint32 of 16 codes, element 0 in
the lowest bits: 64 uint32 = 256 bytes per tile, exactly 2 bits per weight.

With the shared exponent forced, the Blackhole unpacker reads face-row word k of a tile from stored word
(k - 4) mod 64 of that same tile (measured on a P150: tests/perm_probe.py), so each tile's words are stored rotated
left by 4 (ROTATE).
"""

import numpy as np

TILE = 32
FACE = 16
TILE_BYTES = 256
CODE = {-1: 0b11, 0: 0b00, 1: 0b01}
ROTATE = 4


def pack_ternary_tiles(t: np.ndarray) -> np.ndarray:
    """[..., 32, 32] int8 in {-1, 0, 1} -> [..., 64] uint32 tile data."""
    t = np.asarray(t)
    assert t.shape[-2:] == (TILE, TILE), t.shape
    codes = np.where(t > 0, 0b01, np.where(t < 0, 0b11, 0b00)).astype(np.uint32)
    lead = t.shape[:-2]
    # [..., face_r(2), row(16), face_c(2), col(16)] -> [..., face_r, face_c, row, col]
    f = codes.reshape(*lead, 2, FACE, 2, FACE)
    f = np.moveaxis(f, -2, -3)  # [..., 2, 2, 16, 16]
    f = f.reshape(*lead, 4, FACE, FACE)
    shifts = 2 * np.arange(FACE, dtype=np.uint32)
    words = np.bitwise_or.reduce(f << shifts, axis=-1)  # [..., 4, 16]
    words = words.reshape(*lead, 4 * FACE).astype(np.uint32)
    return np.roll(words, -ROTATE, axis=-1)


def unpack_ternary_tiles(words: np.ndarray) -> np.ndarray:
    """Inverse of pack_ternary_tiles: [..., 64] uint32 -> [..., 32, 32] int8."""
    words = np.roll(np.asarray(words, dtype=np.uint32), ROTATE, axis=-1)
    lead = words.shape[:-1]
    w = words.reshape(*lead, 4, FACE, 1)
    codes = (w >> (2 * np.arange(FACE, dtype=np.uint32))) & 3  # [..., 4, 16, 16]
    vals = np.select([codes == 0b01, codes == 0b11], [1, -1], 0).astype(np.int8)
    v = vals.reshape(*lead, 2, 2, FACE, FACE)
    v = np.moveaxis(v, -3, -2)  # [..., 2(face_r), 16(row), 2(face_c), 16(col)]
    return v.reshape(*lead, TILE, TILE)
