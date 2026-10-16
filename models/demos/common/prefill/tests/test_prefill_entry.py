# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.common.prefill.chunk_layout import rotate_chunk_tokens
from models.demos.common.prefill.prefill_entry import (
    EMPTY_ENTRY,
    PrefillEntry,
    pack_forward_tokens,
    pad_entries,
    validate_entries,
)

C, SP = 1024, 8
S = C // SP
EMPTY = EMPTY_ENTRY

GOOD = [
    [PrefillEntry(0, 0, C), PrefillEntry(0, C, 2 * C), PrefillEntry(1, 5 * C, 6 * C), PrefillEntry(2, 0, C)],
    [PrefillEntry(0, 0, C), PrefillEntry(0, C, C + 700), PrefillEntry(1, 4 * C, 5 * C), PrefillEntry(2, 0, 300)],
    [PrefillEntry(0, 2 * C, 3 * C), PrefillEntry(1, 0, 600), EMPTY, EMPTY],
    [
        PrefillEntry(0, 2 * C + 96, 3 * C + 96),
        PrefillEntry(0, 3 * C + 96, 3 * C + 900),
        PrefillEntry(1, C + 512, 2 * C + 512),
        EMPTY,
    ],
]


@pytest.mark.parametrize("entries", GOOD)
def test_validate_entries_accepts(entries):
    validate_entries(entries, chunk_size=C, forward_chunks=4, num_users=4, max_seq_len=8 * C)


BAD = {
    "wrong_count": ([PrefillEntry(0, 0, C)] * 3, "holds 4 entries"),
    "split_run": (
        [PrefillEntry(0, 0, C), PrefillEntry(1, 0, C), PrefillEntry(0, C, 2 * C), EMPTY],
        "two separate runs",
    ),
    "partial_not_last": ([PrefillEntry(0, 0, 500), PrefillEntry(0, C, 2 * C), EMPTY, EMPTY], "must end its slot"),
    "gap_in_run": ([PrefillEntry(0, 0, C), PrefillEntry(0, 2 * C, 3 * C), EMPTY, EMPTY], "does not continue"),
    "overlap_in_run": (
        [PrefillEntry(0, 0, C), PrefillEntry(0, C - 512, 2 * C - 512), EMPTY, EMPTY],
        "does not continue",
    ),
    "empty_not_tail": ([PrefillEntry(0, 0, C), EMPTY, PrefillEntry(1, 0, C), EMPTY], "after an empty one"),
    "all_empty": ([EMPTY] * 4, "at least one non-empty entry"),
    "negative_start": ([PrefillEntry(0, -32, 32), EMPTY, EMPTY, EMPTY], "negative start"),
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


def test_pad_entries(expect_error):
    e = PrefillEntry(1, 0, C)
    assert pad_entries([e], 3) == [e, EMPTY, EMPTY]
    assert pad_entries([e], 1) == [e]
    assert EMPTY.is_empty and not e.is_empty
    with expect_error(AssertionError, "2 entries for a forward of 1"):
        pad_entries([e, e], 1)


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
