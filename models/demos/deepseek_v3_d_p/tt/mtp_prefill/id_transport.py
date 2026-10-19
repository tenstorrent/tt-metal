# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Send the MTP union's token ids down the pipeline instead of its embeddings; the last rank gathers them.

The D2D activation is bf16, so each id travels as three bytes (each exact in bf16) in three rows of one
32-row block per chip. Ids are clamped to ``vocab_size - 1`` first, as the embedding lookup does.
"""

from __future__ import annotations

import ttnn
from models.demos.common.prefill.runners.runner_utils import MTP_ID_ROWS as ID_ROWS

_BYTE_ROWS = 3
_BASES = (1.0, 256.0, 65536.0)


def pack_ids(chunk_ids: ttnn.Tensor, mtp_ids: ttnn.Tensor, *, vocab_size: int, width: int) -> ttnn.Tensor:
    """``[1, 1, L]`` + ``[1, 1, N]`` uint32 ROW_MAJOR ids -> ``[1, 1, 32, width]`` bf16 TILE byte block.

    Neither input is consumed. Requires ``L + N <= width`` and ``vocab_size <= 2**24``.
    """
    n = int(chunk_ids.shape[-1]) + int(mtp_ids.shape[-1])
    assert n <= width, f"{n} ids do not fit one {width}-wide row"
    assert vocab_size <= 1 << 24, f"vocab {vocab_size} does not fit three bytes"
    ids = ttnn.concat([chunk_ids, mtp_ids], dim=-1)
    ids4 = ttnn.reshape(ids, [1, 1, 1, n])
    clamped = ttnn.minimum(ids4, vocab_size - 1)
    ttnn.deallocate(ids)
    tiled = ttnn.to_layout(clamped, ttnn.TILE_LAYOUT)
    ttnn.deallocate(clamped)
    value = ttnn.typecast(tiled, ttnn.float32)
    ttnn.deallocate(tiled)
    rows = []
    for base in reversed(_BASES):
        digit = ttnn.floor(ttnn.multiply(value, 1.0 / base))
        rest = ttnn.subtract(value, ttnn.multiply(digit, base))
        ttnn.deallocate(value)
        rows.append(digit)
        value = rest
    ttnn.deallocate(value)
    rows.reverse()  # byte 0 (least significant) first
    stacked = ttnn.concat(rows, dim=-2)
    for r in rows:
        ttnn.deallocate(r)
    block16 = ttnn.typecast(stacked, ttnn.bfloat16)
    ttnn.deallocate(stacked)
    block = ttnn.pad(block16, [(0, 0), (0, 0), (0, ID_ROWS - _BYTE_ROWS), (0, width - n)], 0.0)
    ttnn.deallocate(block16)
    return block


def unpack_ids(block: ttnn.Tensor, *, num_chunk_ids: int, num_mtp_ids: int) -> tuple:
    """Inverse of :func:`pack_ids` (on the clamped ids): ``(chunk ids [1, 1, L], mtp ids [1, 1, N])`` uint32
    ROW_MAJOR. Does not consume ``block``."""
    n = num_chunk_ids + num_mtp_ids
    s = list(block.shape)
    value = None
    for row, base in enumerate(_BASES):
        digit16 = ttnn.slice(block, [0, 0, row, 0], [s[0], s[1], row + 1, n])
        digit = ttnn.typecast(digit16, ttnn.float32)
        ttnn.deallocate(digit16)
        term = digit if base == 1.0 else ttnn.multiply(digit, base)
        if term is not digit:
            ttnn.deallocate(digit)
        if value is None:
            value = term
        else:
            summed = ttnn.add(value, term)
            ttnn.deallocate(value)
            ttnn.deallocate(term)
            value = summed
    ids_tile = ttnn.typecast(value, ttnn.uint32)
    ttnn.deallocate(value)
    ids = ttnn.to_layout(ids_tile, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(ids_tile)
    ids3 = ttnn.reshape(ids, [1, 1, n])
    chunk = ttnn.slice(ids3, [0, 0, 0], [1, 1, num_chunk_ids])
    mtp = ttnn.slice(ids3, [0, 0, num_chunk_ids], [1, 1, n])
    ttnn.deallocate(ids3)
    return chunk, mtp


__all__ = ["ID_ROWS", "pack_ids", "unpack_ids"]
