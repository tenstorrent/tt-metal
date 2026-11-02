# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lay a ternary weight out the way the DRAM-streaming matmul reads it.

The streaming kernel walks B **column-major in tiles**: for each output column
`n` it reads all `K / 32` tiles of that column consecutively, then moves to
`n + 1`.  Each core owns a contiguous run of `per_core_n` columns inside one DRAM
bank's shard.  So the DRAM image is, per bank::

    for n in this bank's columns:
        for kt in 0..K/32:            <- subblocks of ``subblock_k`` tiles
            subblock of BFP2 tiles, kt-th K tile of column n

and the kernel's per-core byte metadata says how many bytes each subblock is.
That is exactly the layout ``shuffle_tensor_tiles`` produces for the reference
test, but built directly from packed ternary codes instead of from floats.

Two encodings, same tile order:

  A0 (roofline)   codes only -- all tiles bfp2, no scales at all
  A1 (engine)     codes, with the fp16 scale row appended per subblock group

``pack_for_streaming`` writes A0.  ``pack_with_scales`` writes A1 and is what
``ternary_matmul/op.py`` consumes.

N must be padded so every bank gets the same number of columns: the kernel
requires ``per_core_n = n_padded / (num_banks * cores_per_bank)`` to be an
integer, and the padded columns are zeros (all-zero tiles cost their 320 B but
contribute nothing).
"""

from __future__ import annotations

import numpy as np

from .bfp2 import BFP2_TILE_BYTES, TILE, pack_tiles

COLUMNS_PER_BANK_ALIGN = TILE  # N is padded to a multiple of 32 * banks * cores_per_bank


def padded_n(n: int, num_banks: int, cores_per_bank: int = 1) -> int:
    """Smallest N >= ``n`` that splits evenly across banks and cores, tile aligned."""
    step = TILE * num_banks * cores_per_bank
    return ((n + step - 1) // step) * step


def tile_column_major(codes: np.ndarray, n_padded: int, num_banks: int) -> np.ndarray:
    """``codes [N, K]`` -> DRAM tile image ``[num_banks, per_core_n*Kt, 32, 32]``.

    The streaming reader walks one bank's columns in order, and within a column
    all ``Kt`` K-tiles consecutively, so the tile order per bank is
    ``(column, k_tile)``.  ``codes`` is ``[N, K]`` in matmul order
    (``y = x @ W.T``) while the kernel's B is ``[K, N]``, so each tile here is
    the 32x32 block ``codes[n0:n0+32, k0:k0+32]`` read as ``B[k0:k0+32, n0:n0+32].T``
    -- a real transpose, which is why the tile contents are transposed below.
    """
    n, k = codes.shape
    if k % TILE or n_padded % TILE:
        raise ValueError(f"K={k} and N_padded={n_padded} must both be multiples of {TILE}")
    if n_padded < n:
        raise ValueError(f"n_padded={n_padded} is smaller than N={n}")
    per_bank = n_padded // num_banks
    if per_bank * num_banks != n_padded:
        raise ValueError(f"N_padded={n_padded} does not split across {num_banks} banks")

    # Pad output columns (rows of codes) with zeros up to n_padded.
    if n_padded != n:
        pad = np.zeros((n_padded - n, k), dtype=codes.dtype)
        codes = np.concatenate([codes, pad], axis=0)

    kt = k // TILE
    tiles_n = n_padded // TILE
    per_bank_tiles = per_bank // TILE
    # [tile_n, n_in_tile, k_tile, k_in_tile] -> [tile_n, k_tile, k_in_tile, n_in_tile],
    # i.e. the [K, N] orientation the kernel unpacks, tile by tile.
    t = codes.reshape(tiles_n, TILE, kt, TILE).transpose(0, 2, 3, 1)
    # Group output tile-columns by bank, then column-major (column, k_tile).
    t = t.reshape(num_banks, per_bank_tiles, kt, TILE, TILE)
    return np.ascontiguousarray(t.reshape(num_banks, per_bank_tiles * kt, TILE, TILE))


def pack_for_streaming(codes: np.ndarray, num_banks: int, cores_per_bank: int = 1) -> tuple[np.ndarray, int]:
    """A0 layout: every tile as standard 320 B Bfp2_b, no scales.

    Returns ``(bytes [num_banks * per_core_n*Kt * 320], n_padded)``.  The bytes
    are already in the order the reader walks a bank's shard.
    """
    n, k = codes.shape
    n_pad = padded_n(n, num_banks, cores_per_bank)
    tiles = tile_column_major(codes, n_pad, num_banks)
    flat = tiles.reshape(-1, TILE, TILE)
    return pack_tiles(flat).reshape(-1), n_pad


def pack_with_scales(
    codes: np.ndarray, scales: np.ndarray, group: int, num_banks: int, cores_per_bank: int = 1
) -> tuple[np.ndarray, np.ndarray, int]:
    """A1 layout: BFP2 tiles plus one fp16 scale row per subblock.

    The kernel reads one subblock (``subblock_k = group/32`` K tiles) at a time
    and applies the group's scale to that partial product, so the scale has to
    sit right after its subblock in the stream.  Rather than interleave it into
    the tile bytes -- which would break the CB's fixed 320 B page size -- the
    scales go in a second tensor with the same walk order and one fp16 per
    (column, group).

    Returns ``(tile_bytes, scale_bytes, n_padded)``.
    """
    n, k = codes.shape
    if k % group:
        raise ValueError(f"K={k} is not a multiple of the group size {group}")
    n_pad = padded_n(n, num_banks, cores_per_bank)
    subblock_k = group // TILE
    if subblock_k * TILE != group:
        raise ValueError(f"group {group} is not a whole number of {TILE}-wide tiles")

    tiles = tile_column_major(codes, n_pad, num_banks)  # [banks, per_core_n*Kt, 32, 32]
    flat = pack_tiles(tiles.reshape(-1, TILE, TILE)).reshape(-1)

    # Scales: [N, K/group] -> same bank/column-major walk, one fp16 per subblock.
    if n_pad != n:
        pad = np.zeros((n_pad - n, scales.shape[1]), dtype=scales.dtype)
        scales = np.concatenate([scales, pad], axis=0)
    per_bank = n_pad // num_banks
    s = scales.reshape(num_banks, per_bank, scales.shape[1])
    s = np.ascontiguousarray(s.transpose(0, 2, 1))  # [banks, ngroups, per_bank columns]
    return flat, s.astype(np.float16).reshape(-1), n_pad


def subblock_byte_count(subblock_k: int) -> int:
    """Bytes the reader pulls per subblock in the A0/A1 layout (no scales)."""
    return subblock_k * BFP2_TILE_BYTES
