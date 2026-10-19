# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: packed-forward chunk validation, the per-chip MoE padding counts and the packed input layout
(tt/prefill_chunk.py). A forward of F chunks puts each chunk's chunk_local rows back to back on every chip."""

import random

import pytest

from models.demos.common.prefill.chunk_layout import (
    rotate_chunk_tokens,
    rotated_chunk_positions,
    rotated_chunk_real_counts,
)
from models.demos.minimax_m3.tt.prefill_chunk import (
    PrefillChunk,
    pack_forward_tokens,
    packed_real_rows,
    validate_chunks,
)

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
    rng = random.Random(0)
    for _ in range(300):
        chunks = []
        for _ in range(rng.randint(1, 4)):
            start = rng.randrange(0, 4 * C, 32)
            chunks.append(PrefillChunk(len(chunks), start, start + rng.choice([0, 1, 33, S, C // 2, C - 1, C])))
        chunks.sort(key=lambda e: e.is_empty)
        assert packed_real_rows(chunks, SP, S) == brute_real_rows(chunks, SP, S)


BAD = {  # name -> (chunks, expected message)
    "wrong_count": ([PrefillChunk(0, 0, C)] * 3, "holds 4 chunks"),
    "split_run": (
        [PrefillChunk(0, 0, C), PrefillChunk(1, 0, C), PrefillChunk(0, C, 2 * C), EMPTY],
        "two separate runs",
    ),
    "partial_not_last": ([PrefillChunk(0, 0, 500), PrefillChunk(0, C, 2 * C), EMPTY, EMPTY], "must end its slot"),
    "gap_in_run": ([PrefillChunk(0, 0, C), PrefillChunk(0, 2 * C, 3 * C), EMPTY, EMPTY], "does not continue"),
    "empty_not_tail": ([PrefillChunk(0, 0, C), EMPTY, PrefillChunk(1, 0, C), EMPTY], "after an empty one"),
    "all_empty": ([EMPTY] * 4, "at least one non-empty chunk"),
    "negative_start": ([PrefillChunk(0, -32, 32), EMPTY, EMPTY, EMPTY], "negative start"),
    "misaligned_start": ([PrefillChunk(0, 100, 600), EMPTY, EMPTY, EMPTY], "multiple of 32"),
    "more_than_a_chunk": ([PrefillChunk(0, 0, C + 32), EMPTY, EMPTY, EMPTY], "more than one"),
    "cache_overflow": ([PrefillChunk(0, 7 * C + 32, 7 * C + 64), EMPTY, EMPTY, EMPTY], "exceeds the per-user cache"),
    "slot_out_of_range": ([PrefillChunk(4, 0, C), EMPTY, EMPTY, EMPTY], "slot out of range"),
}


@pytest.mark.parametrize("name", BAD)
def test_validate_chunks_rejects(name, expect_error):
    chunks, message = BAD[name]
    with expect_error(AssertionError, message):
        validate_chunks(chunks, chunk_size=C, forward_chunks=4, num_users=4, max_seq_len=8 * C)


@pytest.mark.parametrize("starts", [[0], [0, C, 3 * C, 0], [2 * C + 96, 3 * C + 96, C + 512, 0]])
def test_pack_forward_tokens(starts):
    chunks = [[1000 * k + i for i in range(C)] for k in range(len(starts))]
    packed = pack_forward_tokens(chunks, starts, SP)
    width = len(starts) * S
    for c in range(SP):
        chip = packed[c * width : (c + 1) * width]
        for k, (tokens, start) in enumerate(zip(chunks, starts)):
            assert chip[k * S : (k + 1) * S] == rotate_chunk_tokens(tokens, start, SP)[c * S : (c + 1) * S]
    if len(starts) == 1:
        assert packed == rotate_chunk_tokens(chunks[0], starts[0], SP)
