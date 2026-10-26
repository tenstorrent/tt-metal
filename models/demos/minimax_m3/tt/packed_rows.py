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
    """Per SP chip, the count of leading real rows of the packed forward, for the MoE padding config. A chip
    whose pad rows precede a later real row counts all its rows as real."""
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
        lead = 0
        for k, count in enumerate(counts):
            lead += count[chip]
            if count[chip] < chunk_local:
                if any(later[chip] for later in counts[k + 1 :]):
                    lead = len(chunks) * chunk_local
                break
        rows.append(lead)
    return tuple(rows)


def _chunk_real_counts(slab_offset: int, n_real: int, sp: int, chunk_local: int) -> list:
    if n_real == sp * chunk_local:
        return [chunk_local] * sp
    return rotated_chunk_real_counts(slab_offset, n_real, sp, chunk_local)
