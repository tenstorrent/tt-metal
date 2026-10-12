# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: packed-forward entry validation, the per-chip MoE padding counts and the packed input layout
(tt/prefill_entry.py). A forward of F entries puts each entry's chunk_local rows back to back on every chip."""

import random

import pytest

from models.demos.common.prefill.chunk_layout import (
    rotate_chunk_tokens,
    rotated_chunk_positions,
    rotated_chunk_real_counts,
)
from models.demos.minimax_m3.tt.prefill_entry import (
    PrefillEntry,
    pack_forward_tokens,
    packed_real_rows,
    validate_entries,
)

C, SP = 1024, 8
S = C // SP
EMPTY = PrefillEntry(0, 0, 0)


def brute_real_rows(entries, sp, chunk_local):
    """Leading real rows per chip from the rows' real flags; all rows when a pad row precedes a real one."""
    out = []
    for c in range(sp):
        flags = []
        for e in entries:
            flags += [not e.is_empty and p < e.end for p in rotated_chunk_positions(e.start, sp, chunk_local)[c]]
        n = sum(flags)
        out.append(n if all(flags[:n]) else len(flags))
    return tuple(out)


@pytest.mark.parametrize("sp,chunk_local", [(2, 512), (4, 256), (8, 128), (8, 640)])
def test_packed_real_rows_single_entry_matches_rotated_counts(sp, chunk_local):
    chunk = sp * chunk_local
    for start in range(0, 2 * chunk + 1, 32):
        for n in (1, 31, chunk_local, chunk // 2 + 7, chunk - 1, chunk):
            e = PrefillEntry(0, start, start + n)
            assert packed_real_rows([e], sp, chunk_local) == tuple(rotated_chunk_real_counts(start, n, sp, chunk_local))


CASES = {
    "all_full": [
        PrefillEntry(0, 0, C),
        PrefillEntry(0, C, 2 * C),
        PrefillEntry(1, 5 * C, 6 * C),
        PrefillEntry(2, 0, C),
    ],
    "partial_last": [
        PrefillEntry(0, 0, C),
        PrefillEntry(1, 0, C),
        PrefillEntry(2, 2 * C, 3 * C),
        PrefillEntry(3, 0, 300),
    ],
    "interleaved_partials": [
        PrefillEntry(0, 0, C),
        PrefillEntry(0, C, C + 700),
        PrefillEntry(1, 4 * C, 5 * C),
        PrefillEntry(2, 0, 300),
    ],
    "empty_tail": [PrefillEntry(0, 2 * C, 3 * C), PrefillEntry(1, 0, 600), EMPTY, EMPTY],
    "mid_slab": [
        PrefillEntry(0, 2 * C + 96, 3 * C + 96),
        PrefillEntry(0, 3 * C + 96, 3 * C + 900),
        PrefillEntry(1, C + 512, 2 * C + 512),
        EMPTY,
    ],
}


@pytest.mark.parametrize("name", CASES)
def test_packed_real_rows(name):
    entries = CASES[name]
    validate_entries(entries, chunk_size=C, forward_chunks=4, num_users=4, max_seq_len=8 * C)
    rows = packed_real_rows(entries, SP, S)
    assert rows == brute_real_rows(entries, SP, S)
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
        entries = []
        for _ in range(rng.randint(1, 4)):
            start = rng.randrange(0, 4 * C, 32)
            entries.append(PrefillEntry(len(entries), start, start + rng.choice([0, 1, 33, S, C // 2, C - 1, C])))
        entries.sort(key=lambda e: e.is_empty)
        assert packed_real_rows(entries, SP, S) == brute_real_rows(entries, SP, S)


BAD = {  # name -> (entries, expected message)
    "wrong_count": ([PrefillEntry(0, 0, C)] * 3, "holds 4 entries"),
    "split_run": (
        [PrefillEntry(0, 0, C), PrefillEntry(1, 0, C), PrefillEntry(0, C, 2 * C), EMPTY],
        "two separate runs",
    ),
    "partial_not_last": ([PrefillEntry(0, 0, 500), PrefillEntry(0, C, 2 * C), EMPTY, EMPTY], "must end its slot"),
    "gap_in_run": ([PrefillEntry(0, 0, C), PrefillEntry(0, 2 * C, 3 * C), EMPTY, EMPTY], "does not continue"),
    "empty_not_tail": ([PrefillEntry(0, 0, C), EMPTY, PrefillEntry(1, 0, C), EMPTY], "after an empty one"),
    "all_empty": ([EMPTY] * 4, "at least one non-empty entry"),
    "misaligned_start": ([PrefillEntry(0, 100, 600), EMPTY, EMPTY, EMPTY], "multiple of 32"),
    "more_than_a_chunk": ([PrefillEntry(0, 0, C + 32), EMPTY, EMPTY, EMPTY], "more than one"),
    "cache_overflow": ([PrefillEntry(0, 7 * C + 32, 7 * C + 64), EMPTY, EMPTY, EMPTY], "exceeds the per-user cache"),
    "slot_out_of_range": ([PrefillEntry(4, 0, C), EMPTY, EMPTY, EMPTY], "slot out of range"),
}


@pytest.mark.parametrize("name", BAD)
def test_validate_entries_rejects(name, expect_error):
    entries, message = BAD[name]
    with expect_error(AssertionError, message):
        validate_entries(entries, chunk_size=C, forward_chunks=4, num_users=4, max_seq_len=8 * C)


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
