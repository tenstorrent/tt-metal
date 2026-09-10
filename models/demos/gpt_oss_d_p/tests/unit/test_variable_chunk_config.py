# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for variable-chunk size resolution (no device)."""

from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import resolve_chunk_sizes


def test_single_size_legacy():
    assert resolve_chunk_sizes(8192, (), 57344) == (8192,)


def test_pair_sorted_largest_first():
    assert resolve_chunk_sizes(8192, (1024,), 57344) == (8192, 1024)
    assert resolve_chunk_sizes(1024, (8192,), 57344) == (8192, 1024)


def test_dedupe():
    assert resolve_chunk_sizes(8192, (8192, 1024, 1024), 57344) == (8192, 1024)


def test_128k_pair():
    assert resolve_chunk_sizes(8192, (1024,), 131072) == (8192, 1024)


def test_non_divisible_rejected(expect_error):
    with expect_error(ValueError, "must be a multiple of every supported chunk size"):
        resolve_chunk_sizes(8192, (), 56320)  # 56320 % 8192 != 0
    with expect_error(ValueError, "must be a multiple of every supported chunk size"):
        resolve_chunk_sizes(8192, (1000,), 57344)  # 1000 does not divide 57344


def test_chunk_size_exceeding_max_seq_len_rejected(expect_error):
    with expect_error(ValueError, "must be a multiple of every supported chunk size"):
        resolve_chunk_sizes(8192, (131072,), 57344)
