# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host reference for the counter-based generator behind rand_keyed_tile and dropout_keyed_tile.

Each element's draw is wang(wang(counter ^ key) + key), with counter = ((tile_index * 4 + face) * 8 + row) * 32
+ lane modulo 2^32, key = fmix32(seed) below tile 2^22 and fmix32(fmix32(seed) ^ fmix32(tile_index >> 22)) from it.
"""

import numpy as np


def fmix32(h):
    h &= 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def wang_hash32(x):
    x = (~x) + (x << np.uint32(15))
    x ^= x >> np.uint32(12)
    x = x + (x << np.uint32(2))
    x ^= x >> np.uint32(4)
    x = x + (x << np.uint32(3)) + (x << np.uint32(11))
    x ^= x >> np.uint32(16)
    return x


def reference_draw31(seed, tile_indices):
    """Top 31 bits of each element's draw, laid out as [tiles, 32, 32]."""
    seed_key = fmix32(seed)
    keys = [seed_key if t >> 22 == 0 else fmix32(seed_key ^ fmix32(t >> 22)) for t in np.asarray(tile_indices).tolist()]
    key = np.asarray(keys, dtype=np.uint32)[:, None, None]
    r = np.arange(32)[:, None]
    c = np.arange(32)[None, :]
    face = (r // 16) * 2 + c // 16
    # Within a face, SFPU row d covers face rows 4 * (d // 2) .. + 3 and every other column from d % 2;
    # lane l is face row 4 * (d // 2) + l // 8, column 2 * (l % 8) + d % 2.
    row = 2 * ((r % 16) // 4) + (c % 16) % 2
    lane = ((r % 16) % 4) * 8 + (c % 16) // 2
    t = np.asarray(tile_indices, dtype=np.uint32)[:, None, None]
    counter = ((t * np.uint32(4) + face.astype(np.uint32)) * np.uint32(8) + row.astype(np.uint32)) * np.uint32(
        32
    ) + lane.astype(np.uint32)
    with np.errstate(over="ignore"):
        draw = wang_hash32(wang_hash32(counter ^ key) + key)
    return (draw >> np.uint32(1)).astype(np.int64)


def reference_uniform(seed, tile_indices):
    """rand_keyed_tile with from = 0 and scale = 1: the 31-bit draw rounded to FP32, times 2^-31."""
    return reference_draw31(seed, tile_indices).astype(np.float32) * np.float32(2.0**-31)
