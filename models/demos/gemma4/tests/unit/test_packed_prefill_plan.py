# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host tests for the packed-prefill planner (``plan_packed_prefill_groups``)."""

import pytest

from models.demos.gemma4.tt.generator import plan_packed_prefill_groups
from models.tt_transformers.tt.common import get_padded_prefill_len


@pytest.fixture(autouse=True)
def _blackhole_user_cap(monkeypatch):
    """The planner inherits the batched-prefill user cap (4 on Wormhole, 32 on Blackhole); pin the
    Blackhole value so the groups below do not depend on the host running the test."""
    monkeypatch.setenv("GEMMA4_MAX_BATCHED_PREFILL_USERS", "32")


def test_one_bucket_of_short_prompts_is_one_call():
    groups = plan_packed_prefill_groups([128] * 32, [0] * 32, max_prefill_chunk_size=4096, max_batch_size=32)
    assert groups == [(128, list(range(32)))]


def test_hetero_actual_lengths_share_the_padded_bucket():
    groups = plan_packed_prefill_groups([100, 128, 65, 127], [0] * 4, max_prefill_chunk_size=4096, max_batch_size=32)
    assert groups == [(128, [0, 1, 2, 3])]


def test_buckets_split_and_singles_stay_on_the_stock_path():
    lens = [128, 2048, 128, 300, 128, 300]
    groups = plan_packed_prefill_groups(lens, [0] * 6, max_prefill_chunk_size=4096, max_batch_size=32)
    assert groups == [(128, [0, 2, 4]), (get_padded_prefill_len(300), [3, 5])]  # 2048 alone: stock path


def test_cached_continuations_and_long_prompts_are_excluded():
    lens = [128, 128, 128, 8192, 8192]
    cached = [0, 64, 0, 0, 0]
    groups = plan_packed_prefill_groups(lens, cached, max_prefill_chunk_size=4096, max_batch_size=32)
    assert groups == [(128, [0, 2])]


def test_token_budget_splits_a_wide_bucket():
    # Default budget = the single-pass prefill budget (4096): 32 x 128 is one call, 32 x 1024 is eight calls of 4,
    # 32 x 4096 packs nothing; a 2048 budget splits 32 x 128 into two calls of 16.
    groups = plan_packed_prefill_groups([1024] * 32, [0] * 32, max_prefill_chunk_size=4096, max_batch_size=32)
    assert [len(g[1]) for g in groups] == [4] * 8
    assert groups[0][1] == [0, 1, 2, 3] and groups[-1][1] == [28, 29, 30, 31]
    groups = plan_packed_prefill_groups(
        [128] * 32, [0] * 32, max_prefill_chunk_size=4096, max_batch_size=32, max_tokens=2048
    )
    assert [len(g[1]) for g in groups] == [16, 16]
    assert groups[0][1] == list(range(16)) and groups[1][1] == list(range(16, 32))
    assert plan_packed_prefill_groups([4096] * 32, [0] * 32, max_prefill_chunk_size=4096, max_batch_size=32) == []
    # A raised budget stops at the batched kernel cap (128Ki tokens): 32 x 4096 is two calls of 16.
    groups = plan_packed_prefill_groups(
        [4096] * 32, [0] * 32, max_prefill_chunk_size=4096, max_batch_size=32, max_tokens=1 << 20
    )
    assert [len(g[1]) for g in groups] == [16, 16]


@pytest.mark.parametrize("slots", [[0, 1, 2, 3], [20, 21, 22, 23], [31, 0, 5, 9]])
def test_physical_slots_keep_the_call_within_max_batch(slots):
    groups = plan_packed_prefill_groups(
        [128] * 4, [0] * 4, max_prefill_chunk_size=4096, empty_slots=slots, max_batch_size=32
    )
    assert groups == [(128, [0, 1, 2, 3])]


def test_nothing_to_pack():
    assert plan_packed_prefill_groups([128], [0], max_prefill_chunk_size=4096, max_batch_size=32) == []
    assert plan_packed_prefill_groups([128, 256], [0, 0], max_prefill_chunk_size=4096, max_batch_size=32) == []
