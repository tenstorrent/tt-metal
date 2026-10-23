# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side decode position pad/sentinel handling (no device)."""

import torch
import torch.nn.functional as F


def prepare_rope_and_cache_positions(pos_flat: torch.Tensor, pad_to: int = 32):
    """Mirror Gemma4 prepare_decode_inputs_host position logic."""
    batch = pos_flat.numel()
    pos_i64 = pos_flat.to(torch.int64).clone()
    pos_rope = pos_i64.clone()
    pos_rope[pos_rope < 0] = 0
    pos_rope = pos_rope.reshape(1, batch)
    pos_padded = F.pad(pos_rope, (0, pad_to - batch), "constant", 0) if batch < pad_to else pos_rope
    pos_cache = pos_i64.to(torch.int32)
    return pos_padded, pos_cache


def test_inactive_rows_keep_minus_one_for_cache_but_zero_for_rope():
    # 8 active users + 24 vLLM pad rows at -1 (nearest-bucket pad to 32)
    pos = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80] + [-1] * 24, dtype=torch.int32)
    rope, cache = prepare_rope_and_cache_positions(pos, pad_to=32)
    assert rope.shape == (1, 32)
    assert cache.shape == (32,)
    # RoPE must never see -1 (would become UINT32_MAX and OOB the cos/sin table)
    assert int(rope.min()) >= 0
    assert torch.equal(rope[0, :8], torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int64))
    assert torch.equal(rope[0, 8:], torch.zeros(24, dtype=torch.int64))
    # Cache/SDPA keep -1 skip sentinel on inactive rows
    assert torch.equal(cache[:8], torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int32))
    assert torch.equal(cache[8:], torch.full((24,), -1, dtype=torch.int32))


def test_batch_lt_32_pads_rope_with_zeros():
    pos = torch.tensor([3, 4, -1, -1], dtype=torch.int32)
    rope, cache = prepare_rope_and_cache_positions(pos, pad_to=32)
    assert rope.shape == (1, 32)
    assert torch.equal(rope[0, :4], torch.tensor([3, 4, 0, 0], dtype=torch.int64))
    assert torch.equal(rope[0, 4:], torch.zeros(28, dtype=torch.int64))
    assert torch.equal(cache, torch.tensor([3, 4, -1, -1], dtype=torch.int32))
