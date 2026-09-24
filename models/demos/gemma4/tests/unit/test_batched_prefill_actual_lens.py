# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched prefill gates on identical *padded* buckets (hetero actual OK)."""

from models.tt_transformers.tt.common import get_padded_prefill_len


def can_batch_prefill(prompt_lens, num_cached=None, page_table_ok=True, disable=False):
    """Mirror Gemma4 can_batch_prefill gating after valid_seq_lens pad-fill fix."""
    batch_size = len(prompt_lens)
    num_cached = num_cached or [0] * batch_size
    prefill_seq_lens = [get_padded_prefill_len(int(seq_len) - int(n)) for seq_len, n in zip(prompt_lens, num_cached)]
    return (
        page_table_ok
        and batch_size > 1
        and len(set(prefill_seq_lens)) == 1
        and not disable
        and all(n == 0 for n in num_cached)
    )


def test_same_padded_bucket_different_actual_lens_allows_batch():
    # GPQA-scale prompts all pad to 1024 but differ in real length — batch OK
    # once per-slot valid_seq_lens caps KV fill (attention/prefill.py).
    lens = [154, 159, 184, 165, 380, 231, 191, 478]
    assert len({get_padded_prefill_len(n) for n in lens}) == 1
    assert can_batch_prefill(lens) is True


def test_identical_actual_lens_allows_batch():
    lens = [200, 200, 200, 200]
    assert can_batch_prefill(lens) is True


def test_single_user_never_batches():
    assert can_batch_prefill([200]) is False


def test_different_padded_buckets_disables_batch():
    # 200 → 1024 bucket vs 2000 → 2048 bucket
    assert get_padded_prefill_len(200) != get_padded_prefill_len(2000)
    assert can_batch_prefill([200, 2000]) is False
