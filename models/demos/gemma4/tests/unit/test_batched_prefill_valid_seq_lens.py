# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-slot valid_seq_lens for batched prefill KV fill (host-only)."""

TILE_HEIGHT = 32


def fill_len_for_slot(valid_seq_len, slot_idx, seq_len_per_user, page_len):
    """Mirror attention/prefill.py batched fill length selection."""
    fill_len = seq_len_per_user
    if isinstance(valid_seq_len, (list, tuple)):
        if 0 <= int(slot_idx) < len(valid_seq_len) and valid_seq_len[int(slot_idx)] is not None:
            fill_len = min(fill_len, max(0, int(valid_seq_len[int(slot_idx)])))
    elif valid_seq_len is not None:
        fill_len = min(fill_len, max(0, int(valid_seq_len)))
    fill_len = min(fill_len, page_len)
    tile_end = ((fill_len + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
    tile_end = min(tile_end, seq_len_per_user, page_len) if fill_len > 0 else 0
    return fill_len, tile_end


def test_hetero_actual_lens_tile_ceil_per_slot():
    # batch-32 demo: pad bucket 128, actual lens differ
    last_token_idx = [86, 81, 65, 99]  # actual-1
    valid = [i + 1 for i in last_token_idx]
    page_len = 1024
    seq = 128
    for slot, actual in enumerate(valid):
        fill, tile = fill_len_for_slot(valid, slot, seq, page_len)
        assert fill == actual
        assert tile == ((actual + 31) // 32) * 32
        assert tile <= seq


def test_missing_valid_seq_len_fills_full_pad_bucket():
    fill, tile = fill_len_for_slot(None, 0, 128, 1024)
    assert fill == 128
    assert tile == 128


def test_plan_allows_hetero_actual_with_same_pad():
    from models.demos.gemma4.tests.unit.test_prefill_over_user_cap import plan_metal_prefill_strategy

    # After pad-fill fix, hetero actual + same pad should true-batch / microbatch,
    # not force sequential-only. B=4 → true_batched; B=32 → sequential_global
    # was the *old* plan — update expectation to microbatch/chunk path.
    assert plan_metal_prefill_strategy(4, 128, actual_lens=[80, 90, 70, 100]) == "true_batched"
