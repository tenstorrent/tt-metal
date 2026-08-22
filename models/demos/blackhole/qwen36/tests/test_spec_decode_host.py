# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the spec-decode pure helpers (no device, no checkpoint)."""

import pytest

from models.demos.blackhole.qwen36.tt.spec_decode import BLOCK, block_aligned_prefill_len, greedy_accept


class TestGreedyAccept:
    def test_all_accepted(self):
        m, committed = greedy_accept([5, 6, 7], [5, 6, 7, 8])
        assert m == 3
        assert committed == [5, 6, 7, 8]  # all drafts + bonus

    def test_first_rejected(self):
        m, committed = greedy_accept([5, 6, 7], [9, 6, 7, 8])
        assert m == 0
        assert committed == [9]  # correction only

    def test_middle_rejected(self):
        m, committed = greedy_accept([5, 6, 7], [5, 2, 7, 8])
        assert m == 1
        assert committed == [5, 2]  # accepted prefix + correction

    def test_always_commits_at_least_one(self):
        for targets in ([1, 1, 1, 1], [0, 0, 0, 0]):
            _, committed = greedy_accept([9, 9, 9], targets)
            assert len(committed) >= 1

    def test_target_len_mismatch_asserts(self):
        with pytest.raises(AssertionError):
            greedy_accept([1, 2], [1, 2])


class TestBlockAlignedPrefillLen:
    @pytest.mark.parametrize(
        "prompt_len,expected",
        [
            (1, 0),
            (2, 0),
            (BLOCK, 0),  # strictly below: block-aligned prompt leaves a full-block tail
            (BLOCK + 1, BLOCK),
            (2 * BLOCK, BLOCK),
            (1000, BLOCK * ((1000 - 1) // BLOCK)),
        ],
    )
    def test_values(self, prompt_len, expected):
        a0 = block_aligned_prefill_len(prompt_len)
        assert a0 == expected
        assert a0 % BLOCK == 0
        # The first verify chunk (the prompt tail) is non-empty and <= one block.
        assert 1 <= prompt_len - a0 <= BLOCK
