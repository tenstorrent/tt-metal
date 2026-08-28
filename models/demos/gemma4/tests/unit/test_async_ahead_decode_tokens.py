# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for Gemma4 async-ahead decode token keep / slot_remap safety.

Regresses the LB gemma-4-12B IndexError where nearest-bucket decode changed B
while slot_remap still referred to the previous layout, and mesh-sharded
device buffers were narrower than the host batch.
"""

from __future__ import annotations

import torch

from models.demos.gemma4.tt.async_decode import merge_async_ahead_decode_tokens


def test_matching_buffers_merge_device_token_when_pos_ahead():
    host_toks = torch.tensor([10, 20, 30], dtype=torch.int32)
    host_pos = torch.tensor([5, 6, 7], dtype=torch.int64)
    # Device is one step ahead on slots 0 and 2.
    dev_toks = torch.tensor([11, 20, 31], dtype=torch.int32)
    dev_pos = torch.tensor([6, 6, 8], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos)
    assert src == "merged"
    assert toks.tolist() == [11, 20, 31]
    assert pos.tolist() == [6, 6, 8]


def test_wider_device_feedback_merges_prefix():
    """Gemma4 pads feedback to width 32; host nearest-bucket B may be smaller.

    Slot 0 is async-ahead (pos host+1) → take device token. Remaining padded
    device rows are ignored via ``[:host_b]``.
    """
    host_toks = torch.tensor([99], dtype=torch.int32)
    host_pos = torch.tensor([3], dtype=torch.int64)
    dev_toks = torch.tensor([100, 2, 3, 4], dtype=torch.int32)
    dev_pos = torch.tensor([4, 2, 3, 4], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos)
    assert src == "merged"
    assert toks.tolist() == [100]
    assert pos.tolist() == [4]


def test_wider_device_unrelated_pos_keeps_host():
    """After batch recomposition, slot0 device pos may not match host → host wins."""
    host_toks = torch.tensor([99], dtype=torch.int32)
    host_pos = torch.tensor([3], dtype=torch.int64)
    dev_toks = torch.tensor([100, 2, 3, 4], dtype=torch.int32)
    # Device slot0 at unrelated position — must not clobber host.
    dev_pos = torch.tensor([10, 2, 3, 4], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos)
    assert src == "merged"
    assert toks.tolist() == [99]
    assert pos.tolist() == [3]


def test_sharded_narrow_device_buffer_falls_back():
    """Shard-0 exposes B/num_shards entries; must not index into host_b."""
    host_toks = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    host_pos = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    dev_toks = torch.tensor([10], dtype=torch.int32)  # one shard only
    dev_pos = torch.tensor([1], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos)
    assert src == "host_fallback"
    assert toks.tolist() == [10, 20, 30, 40]


def test_oob_slot_remap_falls_back_no_indexerror():
    """Remap indices from a larger previous batch must not gather."""
    host_toks = torch.tensor([10, 20], dtype=torch.int32)
    host_pos = torch.tensor([5, 6], dtype=torch.int64)
    dev_toks = torch.tensor([11, 21], dtype=torch.int32)
    dev_pos = torch.tensor([6, 7], dtype=torch.int64)
    # Index 3 is OOB for host_b=2 (the pre-fix IndexError path).
    remap = torch.tensor([0, 3], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos, slot_remap_local=remap)
    assert src == "host_fallback"
    assert toks.tolist() == [10, 20]


def test_valid_slot_remap_permutes_device_buffers():
    # Host positions already match the post-condense layout; remapped device
    # buffers are one step ahead so use_dev stays True for every slot.
    host_toks = torch.tensor([10, 20, 30], dtype=torch.int32)
    host_pos = torch.tensor([7, 5, 6], dtype=torch.int64)
    dev_toks = torch.tensor([100, 200, 300], dtype=torch.int32)
    dev_pos = torch.tensor([6, 7, 8], dtype=torch.int64)
    remap = torch.tensor([2, 0, 1], dtype=torch.int64)  # -> toks [300,100,200]

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos, slot_remap_local=remap)
    assert src == "merged"
    assert toks.tolist() == [300, 100, 200]
    assert pos.tolist() == [8, 6, 7]


def test_prefilled_slots_take_host_tokens():
    host_toks = torch.tensor([10, 20], dtype=torch.int32)
    host_pos = torch.tensor([5, 6], dtype=torch.int64)
    dev_toks = torch.tensor([11, 21], dtype=torch.int32)
    dev_pos = torch.tensor([6, 7], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(
        host_toks,
        host_pos,
        dev_toks,
        dev_pos,
        prefilled_local={0},
    )
    assert src == "merged"
    assert toks.tolist() == [10, 21]
    assert pos.tolist() == [5, 7]


def test_mismatched_tok_pos_lengths_fall_back():
    """pos narrower than host_b → fallback (cannot form a full batch)."""
    host_toks = torch.tensor([10, 20], dtype=torch.int32)
    host_pos = torch.tensor([5, 6], dtype=torch.int64)
    dev_toks = torch.tensor([11, 21], dtype=torch.int32)
    dev_pos = torch.tensor([6], dtype=torch.int64)

    toks, _, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos)
    assert src == "host_fallback"
    assert toks.tolist() == [10, 20]


def test_tok_pos_length_mismatch_still_merges_when_both_wide_enough():
    """Sampling writeback may shrink the token buffer's logical shape to 1 while
    RoPE pos stays pad-32. Both are still wide enough for host_b=1 → merge.
    Previous equality check forced host_fallback and async token doubling.
    """
    host_toks = torch.tensor([99], dtype=torch.int32)
    host_pos = torch.tensor([3], dtype=torch.int64)
    # Token buffer reports width 1 (post-sampling logical shape); pos is pad-32.
    dev_toks = torch.tensor([100], dtype=torch.int32)
    dev_pos = torch.tensor([4, 0, 0, 0], dtype=torch.int64)

    toks, pos, src = merge_async_ahead_decode_tokens(host_toks, host_pos, dev_toks, dev_pos)
    assert src == "merged"
    assert toks.tolist() == [100]
    assert pos.tolist() == [4]


def test_slot_remap_beyond_host_batch_indexes_device_rows():
    """A slot id >= host_b is a valid *device row*, not out of bounds.

    Device feedback is padded to width 32 while the host batch can be 1, and a
    request may occupy any slot. The bound was previously ``host_b``, so a
    request living in slot 1 with host_b=1 was rejected as OOB and silently fell
    back to host tokens. The bound is the device buffer width.
    """
    host_toks = torch.tensor([111], dtype=torch.int32)
    host_pos = torch.tensor([100], dtype=torch.int64)
    dev_toks = torch.zeros(32, dtype=torch.int32)
    dev_pos = torch.zeros(32, dtype=torch.int64)
    # slot 0 is stale; slot 1 is this request's real device row.
    dev_toks[0], dev_pos[0] = 999, 7
    dev_toks[1], dev_pos[1] = 222, 100

    merged, merged_pos, src = merge_async_ahead_decode_tokens(
        host_toks, host_pos, dev_toks, dev_pos, slot_remap_local=torch.tensor([1])
    )
    assert src == "merged"
    # Must read slot 1, never slot 0's stale 999.
    assert int(merged[0]) == 222
    assert int(merged_pos[0]) == 100


def test_slot_remap_beyond_device_width_still_falls_back():
    """Past the device buffer width is still OOB and must fall back."""
    host_toks = torch.tensor([111], dtype=torch.int32)
    host_pos = torch.tensor([100], dtype=torch.int64)
    dev_toks = torch.zeros(32, dtype=torch.int32)
    dev_pos = torch.zeros(32, dtype=torch.int64)
    merged, merged_pos, src = merge_async_ahead_decode_tokens(
        host_toks, host_pos, dev_toks, dev_pos, slot_remap_local=torch.tensor([32])
    )
    assert src == "host_fallback"
    assert int(merged[0]) == 111
