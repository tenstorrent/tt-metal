# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The bytes of BFP8 tiles on the host (numpy; no ttnn): the layout the index-key append's row writer relies on
(kernels/attn_key_rows.hpp states it; this file is written from tt_metal/impl/data_format/blockfloat_common.cpp
separately, as arrays with one axis per thing, so that the tests hold the two against each other and both against
the device).

A [32, 32] BFP8 tile is 1,088 bytes: 64 exponent bytes ``[face 4][row 16]`` and 1,024 mantissa bytes
``[face 4][row 16][column 16]`` (sign bit and 7 magnitude bits); faces 0, 1 are tile rows 0..15 (columns 0..15,
16..31), faces 2, 3 tile rows 16..31. A value is ``(-1)^sign * magnitude * 2^(exponent - 127 - 6)``.
A key block is 32 rows x 128 = 4 tiles side by side, 4,352 bytes."""

import numpy as np

TILE_BYTES, TILES, BLOCK_BYTES, BLOCK_ROWS = 1088, 4, 4352, 32


def split(block: np.ndarray):
    """bytes [..., 4352] -> (exponents [..., tile 4, tile row 32, half 2], mantissas [..., 4, 32, 2, 16]),
    reordered to tile rows (copies): half h of tile row r is face 2 (r // 16) + h, row r % 16."""
    tiles = block.reshape(*block.shape[:-1], TILES, TILE_BYTES)
    exps = tiles[..., :64].reshape(*tiles.shape[:-1], 2, 2, 16)  # [face row pair, half, row in face]
    mans = tiles[..., 64:].reshape(*tiles.shape[:-1], 2, 2, 16, 16)
    lead = tuple(range(exps.ndim - 3))
    n = len(lead)
    exps = exps.transpose(*lead, n, n + 2, n + 1).reshape(*tiles.shape[:-1], 32, 2)
    mans = mans.transpose(*lead, n, n + 2, n + 1, n + 3).reshape(*tiles.shape[:-1], 32, 2, 16)
    return exps, mans


def values(block: np.ndarray) -> np.ndarray:
    """bytes [..., 4352] -> the stored values [..., 32, 128] (float64)."""
    exps, mans = split(block)
    sign = np.where(mans & 0x80, -1.0, 1.0)
    mag = (mans & 0x7F).astype(np.float64)
    v = sign * mag * np.exp2(exps.astype(np.float64)[..., None] - 127 - 6)  # [..., tile, row, half, 16]
    lead = tuple(range(v.ndim - 4))
    n = len(lead)
    return v.transpose(*lead, n + 1, n, n + 2, n + 3).reshape(*block.shape[:-1], 32, 128)


def row_bytes(block: np.ndarray, row: int) -> np.ndarray:
    """The bytes that belong to tile row ``row`` of a block [4352]: [tile 4, half 2, 1 exponent + 16 mantissas]."""
    exps, mans = split(block)
    return np.concatenate((exps[:, row, :, None], mans[:, row]), axis=-1).copy()


def put_row(block: np.ndarray, row: int, row_data: np.ndarray) -> np.ndarray:
    """A copy of the block [4352] with tile row ``row`` replaced by ``row_data`` (row_bytes' shape)."""
    out = block.copy()
    tiles = out.reshape(TILES, TILE_BYTES)
    for t in range(TILES):
        for h in range(2):
            face, r = 2 * (row // 16) + h, row % 16
            tiles[t, 16 * face + r] = row_data[t, h, 0]
            start = 64 + 256 * face + 16 * r
            tiles[t, start : start + 16] = row_data[t, h, 1:]
    return out


def other_rows_mask(row: int) -> np.ndarray:
    """bool [4352]: the bytes of a block that do NOT belong to tile row ``row``."""
    probe = np.zeros(BLOCK_BYTES, dtype=np.uint8)
    marked = put_row(probe, row, np.ones((TILES, 2, 17), dtype=np.uint8))
    return marked == 0
