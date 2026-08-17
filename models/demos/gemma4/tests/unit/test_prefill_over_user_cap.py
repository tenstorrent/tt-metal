# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""When batch exceeds the batched-prefill user cap, metal must not remap slots."""

from models.demos.gemma4.tt.generator import max_batched_prefill_users, resolve_batched_prefill_chunk_users


def plan_metal_prefill_strategy(
    batch_size: int,
    prefill_seq_len: int = 128,
    actual_lens=None,
) -> str:
    """Mirror Gemma4Generator.prefill_forward_text batching decision (host-only).

    Hetero actual lengths in one pad bucket are still batched (KV fill capped
    per-slot via ``valid_seq_lens``). B>user_cap uses remapped microbatch chunks.
    """
    del actual_lens  # no longer gates batching once pad-fill is capped per slot
    if batch_size <= 1:
        return "single"
    padded_batch = batch_size
    for b in (1, 2, 4, 8, 16, 32):
        if b >= batch_size:
            padded_batch = b
            break
    max_users = resolve_batched_prefill_chunk_users(padded_batch, prefill_seq_len)
    if batch_size > max_users:
        return "microbatch_remapped_slots"
    return "true_batched"


def test_user_cap_default_is_four():
    assert max_batched_prefill_users() == 4


def test_batch_gt_user_cap_uses_microbatch():
    assert plan_metal_prefill_strategy(8) == "microbatch_remapped_slots"
    assert plan_metal_prefill_strategy(32) == "microbatch_remapped_slots"


def test_hetero_actual_lens_same_pad_bucket_still_batches():
    # Pad-fill fix: hetero actual + same pad bucket stays on batched/microbatch path.
    lens = [87, 82, 80, 66, 100, 91, 83, 78]
    assert plan_metal_prefill_strategy(8, prefill_seq_len=128, actual_lens=lens) == "microbatch_remapped_slots"
    assert plan_metal_prefill_strategy(4, prefill_seq_len=128, actual_lens=lens[:4]) == "true_batched"


def test_batch_le_user_cap_uses_true_batched():
    assert plan_metal_prefill_strategy(1) == "single"
    assert plan_metal_prefill_strategy(2) == "true_batched"
    assert plan_metal_prefill_strategy(4) == "true_batched"
