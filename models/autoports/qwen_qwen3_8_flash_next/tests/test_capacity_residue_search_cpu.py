# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure CPU planner tests; no TTNN import or hardware access."""

from models.autoports.qwen_qwen3_8_flash_next.tests.capacity_residue_search import search_residue


def test_independent_tail_boundaries():
    limits = {27: 2523, 29: 2557, 30: 2590, 32: 3712}
    for residue, limit in limits.items():
        result = search_residue(lambda length: length <= limits[length % 32 or 32], residue, 32, 3712)
        assert result["largest_passing_candidate"] == limit
        assert result["next_observed_failure"] == (None if limit == 3712 else limit + 32)
        lengths = [row["length"] for row in result["observations"]]
        assert len(set(lengths)) == len(lengths)
        assert all(length % 32 == residue % 32 and 32 <= length <= 3712 for length in lengths)
        assert result["candidate_only"] and result["monotonicity_assumed_not_proven"]


def test_failed_minimum_does_not_invent_a_pass():
    result = search_residue(lambda length: False, 27, 32, 3712)
    assert result["largest_passing_candidate"] is None
    assert result["next_observed_failure"] == 59
    assert result["stopped_at_failed_minimum"]
    assert len(result["observations"]) == 1


def test_single_block_and_unaligned_bounds():
    result = search_residue(lambda length: True, 29, 90, 95)
    assert result["largest_passing_candidate"] == 93
    assert result["next_observed_failure"] is None
    assert result["observations"] == [{"block_index": 2, "length": 93, "passed": True}]
