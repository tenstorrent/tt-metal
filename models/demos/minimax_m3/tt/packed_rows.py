# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real rows per SP chip of a packed prefill forward, for the MoE padding config.

A forward of ``forward_tokens_budget`` rows holds ``forward_tokens_budget // chunk_size`` chunks
(common/prefill/prefill_chunk.py) in pack order. Each SP chip holds every chunk's ``chunk_size // sp`` rows back
to back, in the chunk's rotated per-chip layout. Positions increase with the local row, so on every chip each
chunk's real rows lead its ``chunk_local``-row block, rotated or not (rotated_chunk_real_counts).
"""

from functools import lru_cache

from models.demos.common.prefill.chunk_layout import rotated_chunk_real_counts


def packed_real_rows(chunks, sp: int, chunk_local: int) -> tuple:
    """Per SP chip, the rows up to and including its last real row, for the MoE padding config (which takes one
    leading count per chip). It masks the pad rows after a chip's last real row (a final partial chunk's, or empty
    chunks'). The pad rows before a later real row (a partial chunk followed by a non-empty one) fall inside the
    count: packed_real_row_mask marks them for a gate that masks per row."""
    return _packed_real_rows(tuple(chunks), sp, chunk_local)


@lru_cache(maxsize=1024)  # every MoE layer asks for the same forward
def _packed_real_rows(chunks: tuple, sp: int, chunk_local: int) -> tuple:
    counts = _counts(chunks, sp, chunk_local)
    rows = []
    for chip in range(sp):
        lead = 0
        for k, count in enumerate(counts):
            if count[chip]:
                lead = k * chunk_local + count[chip]
        rows.append(lead)
    return tuple(rows)


def packed_real_row_mask(chunks, sp: int, chunk_local: int) -> tuple:
    """Per SP chip, one flag per row of the packed forward: 1 for a real row, 0 for a pad row. Each chunk's flags on
    a chip are a run of ones (its real rows there) then zeros, rotated chunk or not."""
    return _packed_real_row_mask(tuple(chunks), sp, chunk_local)


@lru_cache(maxsize=1024)
def _packed_real_row_mask(chunks: tuple, sp: int, chunk_local: int) -> tuple:
    counts = _counts(chunks, sp, chunk_local)
    return tuple(tuple(int(r < count[chip]) for count in counts for r in range(chunk_local)) for chip in range(sp))


def packed_routed_pad_rows(chunks, sp: int, chunk_local: int) -> int:
    """Pad rows packed_real_rows counts as real because they precede a later real row on their chip: the rows only
    packed_real_row_mask masks. Zero when every partial chunk is the last non-empty one on its chips."""
    chunks = tuple(chunks)
    counts = _counts(chunks, sp, chunk_local)
    lead = _packed_real_rows(chunks, sp, chunk_local)
    return sum(lead[chip] - sum(count[chip] for count in counts) for chip in range(sp))


def _counts(chunks: tuple, sp: int, chunk_local: int) -> list:
    """Per chunk, per SP chip, its real rows (none for an empty chunk)."""
    return [
        _chunk_real_counts(c.actual_start % (sp * chunk_local), c.actual_end - c.actual_start, sp, chunk_local)
        for c in chunks
    ]


def _chunk_real_counts(slab_offset: int, n_real: int, sp: int, chunk_local: int) -> list:
    if n_real == sp * chunk_local:
        return [chunk_local] * sp
    return rotated_chunk_real_counts(slab_offset, n_real, sp, chunk_local)
