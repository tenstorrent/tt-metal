# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P1c-B no-device tests: the paged lane config, the paged lane constants' host rows and the driver's host page
allocator / page-table row builder."""

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import (
    Qwen38PageAllocator,
    lane_page_count,
    page_table_rows,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging

CONTEXT = 8192


@pytest.mark.parametrize(
    "paging, message",
    [
        (Qwen38LanePaging(48, 10, 8 * 22), "power of two"),
        (Qwen38LanePaging(16, 10, 8 * 64), "power of two"),
        (Qwen38LanePaging(64, 10, 130), "multiple of 8"),
        (Qwen38LanePaging(64, 10, 64), "cover the allocated context"),
        (Qwen38LanePaging(64, 1, 128), "num_pages"),
    ],
)
def test_paging_validation_rejects(paging, message, expect_error):
    with expect_error(ValueError, message):
        paging.validate(CONTEXT)


def test_paging_validation_accepts_and_sizes_the_pool():
    paging = Qwen38LanePaging(64, 40, 128).validate(CONTEXT)
    assert paging.pool_rows == 40 * 64
    assert Qwen38LanePaging(32, 2, 256).validate(CONTEXT).pool_rows == 64


def test_paged_lane_constants_carry_zero_lane_offsets():
    for lanes in (2, 8, 32):
        assert qsa.lane_kv_offsets(lanes, CONTEXT, paged=True) == [0] * qsa.MAX_LANES
        assert qsa.lane_kv_offsets(lanes, CONTEXT)[1] == CONTEXT
        paged = qsa.qsa_lane_constant_rows(lanes, CONTEXT, paged=True)
        plain = qsa.qsa_lane_constant_rows(lanes, CONTEXT)
        assert not paged["kv_offsets_row"].any()
        assert torch.equal(paged["block_offsets_lanes"][0, 0, 0], plain["block_offsets_lanes"][0, 0, 0])
        assert torch.equal(paged["block_offsets_lanes"][0, 0, 5], plain["block_offsets_lanes"][0, 0, 0])
        assert torch.equal(paged["arange_slots_lanes"][0, 0, 7], torch.arange(qsa.SPARSE_INDEX_CAPACITY))
        # the unpaged rows are unchanged: lane u = lane 0's row + u * C
        assert plain["arange_slots_lanes"][0, 0, 1, 0] == CONTEXT
        for name in ("arange32_tile", "lane_indicator"):
            assert torch.equal(paged[name], plain[name])


def test_lane_page_count():
    assert lane_page_count(300, 32, 64) == 6  # 333 rows
    assert lane_page_count(31, 32, 64) == 1  # 64 rows
    assert lane_page_count(32, 32, 64) == 2  # 65 rows
    assert lane_page_count(300, 32, 32) == 11


def test_allocator_hands_out_distinct_shuffled_pages_and_never_page_zero():
    allocator = Qwen38PageAllocator(200, seed=3)
    lanes = [allocator.allocate(n) for n in (6, 9, 12, 15)]
    flat = [page for pages in lanes for page in pages]
    assert len(flat) == len(set(flat)) == 42 and 0 not in flat and all(1 <= p < 200 for p in flat)
    assert any(b - a != 1 for pages in lanes for a, b in zip(pages, pages[1:])), "pages must not be contiguous"
    assert Qwen38PageAllocator(200, seed=3).allocate(6) == lanes[0], "seeded: deterministic"
    assert Qwen38PageAllocator(200, seed=4).allocate(6) != lanes[0]
    # interleaved: the lanes' page ids overlap in range
    assert max(lanes[0]) > min(lanes[1]) and max(lanes[1]) > min(lanes[0])


def test_allocator_exhaustion_release_and_double_free(expect_error):
    allocator = Qwen38PageAllocator(5, seed=0)
    pages = allocator.allocate(4)
    assert sorted(pages) == [1, 2, 3, 4]
    with expect_error(RuntimeError, "cannot allocate"):
        allocator.allocate(1)
    allocator.release(pages[:2])
    again = allocator.allocate(2)
    assert sorted(again) == sorted(pages[:2])
    with expect_error(RuntimeError, "not allocated"):
        allocator.release([0])
    allocator.release(again)
    with expect_error(RuntimeError, "not allocated"):
        allocator.release(again)
    with expect_error(ValueError):
        Qwen38PageAllocator(1)


def test_page_table_rows_keep_every_lane_row(expect_error):
    rows = [[3, 1], [], [7]]
    out = page_table_rows(rows, 3)
    assert out == rows and out is not rows and out[0] is not rows[0]
    with expect_error(ValueError):
        page_table_rows(rows, 4)
