# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cheap invariants for the throwaway fixed-5K offset topology."""

from __future__ import annotations

import pytest

from models.demos.deepseek_v3_d_p.tt.kda.offset_prototype import offset_segments


@pytest.mark.parametrize("actual_start", range(0, 5120, 32))
def test_offset_segments_cover_each_physical_row_once(actual_start: int) -> None:
    segments = offset_segments(actual_start, sp_size=8, local_sequence=640)
    rows = [(rank, row) for rank, begin, end in segments for row in range(begin, end)]

    assert len(rows) == 5120
    assert len(set(rows)) == 5120
    assert set(rows) == {(rank, row) for rank in range(8) for row in range(640)}
    assert all(begin < end and (end - begin) % 32 == 0 for _, begin, end in segments)


@pytest.mark.parametrize("actual_start", range(0, 5120, 640))
def test_device_boundary_has_zero_tail_and_rotated_rank_order(actual_start: int) -> None:
    segments = offset_segments(actual_start, sp_size=8, local_sequence=640)
    boundary = actual_start // 640

    assert segments == tuple(((boundary + step) % 8, 0, 640) for step in range(8))


def test_offset_960_splits_sp1_between_head_and_tail() -> None:
    assert offset_segments(960, sp_size=8, local_sequence=640) == (
        (1, 0, 320),
        (2, 0, 640),
        (3, 0, 640),
        (4, 0, 640),
        (5, 0, 640),
        (6, 0, 640),
        (7, 0, 640),
        (0, 0, 640),
        (1, 320, 640),
    )


@pytest.mark.parametrize("actual_start", [-32, 1, 639])
def test_offset_rejects_invalid_alignment(actual_start: int, expect_error) -> None:
    with expect_error(ValueError, "non-negative multiple"):
        offset_segments(actual_start, sp_size=8, local_sequence=640)
