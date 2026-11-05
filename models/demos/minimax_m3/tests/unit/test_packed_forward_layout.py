# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the per-chip MoE padding counts of a packed forward (tt/packed_rows.py). A forward of F chunks puts
each chunk's chunk_local rows back to back on every chip."""

import pytest
import torch

from models.demos.common.prefill.chunk_layout import rotated_chunk_positions, rotated_chunk_real_counts
from models.demos.common.prefill.prefill_chunk import PrefillChunk, validate_chunks
from models.demos.minimax_m3.tt.packed_rows import packed_real_rows

C, SP = 1024, 8
S = C // SP
EMPTY = PrefillChunk(0, 0, 0)


def brute_real_rows(chunks, sp, chunk_local):
    """Leading real rows per chip from the rows' real flags; all rows when a pad row precedes a real one."""
    out = []
    for c in range(sp):
        flags = []
        for e in chunks:
            flags += [
                not e.is_empty and p < e.actual_end for p in rotated_chunk_positions(e.actual_start, sp, chunk_local)[c]
            ]
        n = sum(flags)
        out.append(n if all(flags[:n]) else len(flags))
    return tuple(out)


@pytest.mark.parametrize("sp,chunk_local", [(2, 512), (4, 256), (8, 128), (8, 640)])
def test_packed_real_rows_single_entry_matches_rotated_counts(sp, chunk_local):
    chunk = sp * chunk_local
    for start in range(0, 2 * chunk + 1, 32):
        for n in (1, 31, chunk_local, chunk // 2 + 7, chunk - 1, chunk):
            e = PrefillChunk(0, start, start + n)
            assert packed_real_rows([e], sp, chunk_local) == tuple(rotated_chunk_real_counts(start, n, sp, chunk_local))


CASES = {
    "all_full": [
        PrefillChunk(0, 0, C),
        PrefillChunk(0, C, 2 * C),
        PrefillChunk(1, 5 * C, 6 * C),
        PrefillChunk(2, 0, C),
    ],
    "partial_last": [
        PrefillChunk(0, 0, C),
        PrefillChunk(1, 0, C),
        PrefillChunk(2, 2 * C, 3 * C),
        PrefillChunk(3, 0, 300),
    ],
    "interleaved_partials": [
        PrefillChunk(0, 0, C),
        PrefillChunk(0, C, C + 700),
        PrefillChunk(1, 4 * C, 5 * C),
        PrefillChunk(2, 0, 300),
    ],
    "empty_tail": [PrefillChunk(0, 2 * C, 3 * C), PrefillChunk(1, 0, 600), EMPTY, EMPTY],
    "mid_slab": [
        PrefillChunk(0, 2 * C + 96, 3 * C + 96),
        PrefillChunk(0, 3 * C + 96, 3 * C + 900),
        PrefillChunk(1, C + 512, 2 * C + 512),
        EMPTY,
    ],
}


@pytest.mark.parametrize("name", CASES)
def test_packed_real_rows(name):
    chunks = CASES[name]
    validate_chunks(chunks, chunk_size=C, forward_chunks=4, num_users=4, max_seq_len=8 * C)
    rows = packed_real_rows(chunks, SP, S)
    assert rows == brute_real_rows(chunks, SP, S)
    if name == "all_full":
        assert rows == (4 * S,) * SP
    if name == "partial_last":
        assert rows == tuple(3 * S + r for r in rotated_chunk_real_counts(0, 300, SP, S))
    if name == "empty_tail":
        assert rows == tuple(S + r for r in rotated_chunk_real_counts(0, 600, SP, S))  # empties are pad: no route-all
    if name == "interleaved_partials":
        assert 4 * S in rows  # chips whose pad rows precede a later real row route all rows


def test_packed_real_rows_random():
    gen = torch.Generator().manual_seed(0)
    lengths = [0, 1, 33, S, C // 2, C - 1, C]

    def randint(lo, hi):  # inclusive
        return int(torch.randint(lo, hi + 1, (1,), generator=gen))

    for _ in range(300):
        chunks = []
        for _ in range(randint(1, 4)):
            start = 32 * randint(0, 4 * C // 32 - 1)
            chunks.append(PrefillChunk(len(chunks), start, start + lengths[randint(0, len(lengths) - 1)]))
        chunks.sort(key=lambda e: e.is_empty)
        assert packed_real_rows(chunks, SP, S) == brute_real_rows(chunks, SP, S)
