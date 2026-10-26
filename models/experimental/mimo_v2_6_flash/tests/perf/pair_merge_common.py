# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared setup of the KV pair split's decode merge region (tt/pair_split.PairSplit.merge_heads) on Galaxy row 0:
from a global layer's SDPA outputs (each chip's partial o [1, B, 16, 128] bf16 in L1 and its LSE [1, B, 16, 1] fp32)
to the o projection's input [1, 1, B, 2048] bf16 in L1, with the pair's merge weight applied
(reference/kv_pair_split.py). Used by kagent_pair_merge_bench.py and kagent_pair_merge_check.py."""

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.reference import kv_pair_split as kps
from models.experimental.mimo_v2_6_flash.tests.perf.moe_decode_common import close, open_row  # noqa: F401
from models.experimental.mimo_v2_6_flash.tt import kv as kv_lib
from models.experimental.mimo_v2_6_flash.tt.attention import ConcatHeads
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan
from models.experimental.mimo_v2_6_flash.tt.pair_split import PairSplit

TP = 8
HEADS = 16
DV = 128
MAX_BATCH = 32
LAYERS = 9  # global layers per decode step: one merge each


def build(row):
    plan = MeshPlan(row)
    return plan, PairSplit(plan, MAX_BATCH), ConcatHeads(plan, HEADS, DV)


def host_inputs(users, seed):
    """Per-chip partials and LSEs as a global layer's SDPA leaves them: LSEs around 20 (scale * q.k plus log of the
    keys), the pair's two differing by a few units, and three special rows per chip: user 0 sees no key on the even
    chip, user 1 none on the odd chip (that chip's o is 0 and its LSE kv_pair_split.NO_KEY_LSE), user 2 is idle (both
    chips). Returns (o [TP, users, 16, 128] bf16, lse [TP, users, 16, 1] fp32)."""
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(TP, users, HEADS, DV, generator=g).bfloat16()
    lse = 20.0 + 4.0 * torch.randn(TP, users, HEADS, 1, generator=g)
    lse[1::2] = lse[0::2] + 3.0 * torch.randn(TP // 2, users, HEADS, 1, generator=g)  # partners close by
    for chips, user in ((slice(0, None, 2), 0), (slice(1, None, 2), 1), (slice(None), 2)):
        if user < users:
            lse[chips, user] = kps.NO_KEY_LSE
            o[chips, user] = 0
    return o, lse.float()


def device_inputs(plan, o, lse):
    memory = ttnn.L1_MEMORY_CONFIG
    o_tt = kv_lib.per_chip(plan, o.unsqueeze(1), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    lse_tt = kv_lib.per_chip(plan, lse.unsqueeze(1), ttnn.float32, ttnn.TILE_LAYOUT)
    return ttnn.to_memory_config(o_tt, memory), ttnn.to_memory_config(lse_tt, memory)


def reference(o, lse):
    """Expected per chip [TP, users, 2048] in fp64: alpha * o with alpha from the chip's and its partner's LSE, heads
    concatenated (head h at columns h * 128 .. h * 128 + 127)."""
    partner = torch.arange(TP) ^ 1
    alpha = kps.merge_weights(lse.double(), lse[partner].double())
    return (alpha * o.double()).reshape(TP, o.shape[1], HEADS * DV)


def run(pair, concat, o_tt, lse_tt):
    return pair.merge_heads(o_tt, lse_tt, concat)


def to_host(out, users):
    """[TP, users, 2048] float32 from the per-chip [1, 1, B, 2048] outputs."""
    chips = [ttnn.to_torch(t).float().reshape(-1, HEADS * DV)[:users] for t in ttnn.get_device_tensors(out)]
    return torch.stack(chips)
