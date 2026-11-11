# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real rows per SP chip of a packed prefill forward, for the MoE padding config.

A forward of ``forward_tokens_budget`` rows holds ``forward_tokens_budget // chunk_size`` chunks
(common/prefill/prefill_chunk.py) in pack order. Each SP chip holds every chunk's ``chunk_size // sp`` rows back
to back, in the chunk's rotated per-chip layout.
"""

from functools import lru_cache

from models.demos.common.prefill.chunk_layout import rotated_chunk_real_counts


def packed_real_rows(chunks, sp: int, chunk_local: int) -> tuple:
    """Per SP chip, the rows up to and including its last real row, for the MoE padding config (which takes one
    leading count per chip). Pad rows before a later real row on the chip are routed like real rows; pad rows
    after its last real row (a final partial chunk's, or empty chunks') are not. Routing those trailing rows would
    make the MoE's per-expert token counts, and so its output, depend on what they hold, which differs between a
    traced forward (an empty chunk runs against a scratch slot) and an eager one (it is skipped)."""
    # TODO: per-chunk real-row ranges in the MoE padding config, so a partial chunk followed by a non-empty one
    # does not route its pad rows.
    return _packed_real_rows(tuple(chunks), sp, chunk_local)


@lru_cache(maxsize=1024)  # every MoE layer asks for the same forward
def _packed_real_rows(chunks: tuple, sp: int, chunk_local: int) -> tuple:
    counts = [
        _chunk_real_counts(c.actual_start % (sp * chunk_local), c.actual_end - c.actual_start, sp, chunk_local)
        for c in chunks
    ]
    rows = []
    for chip in range(sp):
        # Each chunk is right-padded on every chip, so its real rows lead its chunk_local-row block.
        lead = 0
        for k, count in enumerate(counts):
            if count[chip]:
                lead = k * chunk_local + count[chip]
        rows.append(lead)
    return tuple(rows)


def packed_routed_pad_rows(chunks, sp: int, chunk_local: int) -> int:
    """Pad rows the MoE routes in a packed forward: those packed_real_rows counts as real because they precede a later
    real row on their chip. Zero when every partial chunk is the last non-empty one on its chips."""
    chunks = tuple(chunks)
    counts = [
        _chunk_real_counts(c.actual_start % (sp * chunk_local), c.actual_end - c.actual_start, sp, chunk_local)
        for c in chunks
    ]
    lead = _packed_real_rows(chunks, sp, chunk_local)
    return sum(lead[chip] - sum(count[chip] for count in counts) for chip in range(sp))


def _chunk_real_counts(slab_offset: int, n_real: int, sp: int, chunk_local: int) -> list:
    if n_real == sp * chunk_local:
        return [chunk_local] * sp
    return rotated_chunk_real_counts(slab_offset, n_real, sp, chunk_local)
