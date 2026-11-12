# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Choice of the decode trace row counts and of the trace for a step (host only)."""

import pytest

from models.demos.gemma4_31b_qb2.tt.decode_rows import ROW_SIZES, decode_row_sizes, rows_covering


@pytest.mark.parametrize(
    "slots, expected",
    [(1, ()), (2, (1,)), (4, (1,)), (5, (1, 4)), (8, (1, 4)), (16, (1, 4, 8)), (32, (1, 4, 8, 16)), (12, (1, 4, 8))],
)
def test_extra_trace_sizes_are_the_standard_ones_below_the_slot_count(slots, expected):
    assert decode_row_sizes(slots, 0) == expected


def test_speculative_server_keeps_its_single_trace():
    assert decode_row_sizes(4, 3) == ()
    assert decode_row_sizes(32, 1) == ()


@pytest.mark.parametrize(
    "live, expected",
    [(0, 1), (1, 1), (2, 4), (4, 4), (5, 8), (8, 8), (9, 16), (16, 16), (17, 32), (32, 32)],
)
def test_smallest_covering_trace(live, expected):
    assert rows_covering(live, decode_row_sizes(32, 0), 32) == expected


def test_a_small_server_falls_back_to_its_full_trace():
    sizes = decode_row_sizes(6, 0)
    assert sizes == (1, 4)
    assert rows_covering(5, sizes, 6) == 6
    assert rows_covering(4, sizes, 6) == 4


def test_sizes_are_sorted_and_the_default_set_is_fixed():
    assert ROW_SIZES == tuple(sorted(ROW_SIZES)) == (1, 4, 8, 16, 32)
