# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for dense-SP gather-buffer sizing."""

from models.demos.gpt_oss_d_p.tt.attention.dense_sp import _gather_seq_len


def test_gather_seq_len_full_attention_uses_full_sequence():
    assert _gather_seq_len(None, 1024, 8192) == 8192


def test_gather_seq_len_sliding_attention_reserves_two_halos():
    assert _gather_seq_len(128, 1024, 8192) == 2048


def test_gather_seq_len_does_not_exceed_short_sequence():
    assert _gather_seq_len(128, 1024, 1024) == 1024
