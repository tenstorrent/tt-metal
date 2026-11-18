# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Choice of the decode trace row counts and of the trace for a step (host only)."""

import pytest

from models.demos.gemma4_31b_qb2.tt.decode_rows import (
    ROW_SIZES,
    decode_row_sizes,
    rows_covering,
    verify_shapes,
    verify_users,
)


@pytest.mark.parametrize(
    "slots, expected",
    [(1, ()), (2, (1,)), (4, (1,)), (5, (1, 4)), (8, (1, 4)), (16, (1, 4, 8)), (32, (1, 4, 8, 16)), (12, (1, 4, 8))],
)
def test_extra_trace_sizes_are_the_standard_ones_below_the_slot_count(slots, expected):
    assert decode_row_sizes(slots) == expected


@pytest.mark.parametrize(
    "slots, k, expected",
    [
        (32, 5, ((1, 5), (4, 3))),
        (4, 5, ((1, 5), (4, 3))),
        (2, 5, ((1, 5), (2, 3))),
        (1, 5, ((1, 5),)),
        (32, 2, ((1, 2), (4, 2))),
        (32, 0, ()),
    ],
)
def test_verify_traces_follow_the_users_and_the_requested_draft_length(slots, k, expected):
    assert verify_shapes(slots, k) == expected


def test_every_verify_trace_fits_the_decode_tile():
    for slots in (1, 2, 4, 8, 32):
        for k in range(1, 8):
            assert all(users * (1 + kk) <= 32 for users, kk in verify_shapes(slots, k))


@pytest.mark.parametrize(
    "live, expected",
    [(0, 1), (1, 1), (2, 4), (4, 4), (5, 8), (8, 8), (9, 16), (16, 16), (17, 32), (32, 32)],
)
def test_smallest_covering_trace(live, expected):
    assert rows_covering(live, decode_row_sizes(32), 32) == expected


def test_a_small_server_falls_back_to_its_full_trace():
    sizes = decode_row_sizes(6)
    assert sizes == (1, 4)
    assert rows_covering(5, sizes, 6) == 6
    assert rows_covering(4, sizes, 6) == 4


def test_sizes_are_sorted_and_the_default_set_is_fixed():
    assert ROW_SIZES == tuple(sorted(ROW_SIZES)) == (1, 4, 8, 16, 32)


@pytest.mark.parametrize("live, expected", [(1, 1), (2, 4), (4, 4), (5, None), (8, None), (32, None)])
def test_verify_users_is_none_when_more_requests_are_live_than_any_verify_trace_covers(live, expected):
    assert verify_users(live, verify_shapes(32, 5)) == expected
