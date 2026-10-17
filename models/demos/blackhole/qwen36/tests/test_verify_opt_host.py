# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the verify-path optimizations (QWEN36_VERIFY_SHARDED_ARGMAX, QWEN36_VERIFY_STAGING_CACHE).
No device.

Run: pytest models/demos/blackhole/qwen36/tests/test_verify_opt_host.py -q
"""
import torch

TIE_SENTINEL = 1 << 24  # Qwen36Model._DL_TIE_SENTINEL


def sharded_argmax_ref(logits, n):
    """Torch replica of the device recipe: per-shard first-index argmax + max (shard_argmax_max_rows), then
    _dl_greedy_token_row_batched's combine: global max, +sentinel on shards strictly below it, min global index."""
    R, V = logits.shape
    per = V // n
    shards = logits.reshape(R, n, per)
    local_val, local_idx = shards.max(dim=-1)  # torch.max(dim) returns the first max index on CPU
    local_idx = torch.stack([torch.argmax(shards[:, d], dim=-1) for d in range(n)], dim=1)
    gidx = local_idx.to(torch.int32) + (torch.arange(n, dtype=torch.int32) * per)[None]
    gmax = local_val.max(dim=1, keepdim=True).values
    cand = gidx + (local_val < gmax).to(torch.int32) * TIE_SENTINEL
    return cand.min(dim=1).values


def test_sharded_argmax_matches_full_argmax_random():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(12, 8 * 64, generator=g).to(torch.bfloat16)
    assert torch.equal(sharded_argmax_ref(logits, 8).to(torch.int64), torch.argmax(logits, dim=-1))


def test_sharded_argmax_ties_pick_lowest_global_index():
    # bf16 logits tie constantly; force ties within a shard and across shards.
    n, per = 8, 16
    logits = torch.full((4, n * per), -1.0, dtype=torch.bfloat16)
    logits[0, [5, 9]] = 3.0  # same shard
    logits[1, [per * 2 + 1, per * 6 + 3]] = 2.0  # different shards: lowest index (shard 2) wins
    logits[2, :] = 0.0  # all tied -> index 0
    logits[3, [per * 7, per * 7 + 1, per * 4 + 15]] = 1.0  # shard 4's last beats shard 7's
    ref = torch.argmax(logits, dim=-1)
    assert ref.tolist() == [5, per * 2 + 1, 0, per * 4 + 15]
    assert torch.equal(sharded_argmax_ref(logits, n).to(torch.int64), ref)
