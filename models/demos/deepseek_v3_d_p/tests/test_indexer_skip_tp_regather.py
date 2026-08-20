# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Locks the gating predicate for the indexer top-k TP-regather skip (`skip_tp_regather`).

Why this test exists: the skip is *value-preserving* (gather-then-partition is an identity over
the TP axis), so the full-model PCC/correctness matrix keeps passing even if the optimization is
accidentally disabled — the perf win would regress with nothing failing. `ttMLA._needs_head_to_seq_reshard`
is the single decision that gates the skip (and is passed straight into `TtIndexer(skip_tp_regather=...)`),
so pinning its truth table here catches a silent regression cheaply and without a device.

The rule: the skip fires exactly when the per-chip head shard is too thin for `sparse_sdpa`
(needs `H/tp >= 32` and `H/tp % 32 == 0`) — i.e. when `_sparse_mla` will transpose heads->seq and
re-split the indices over TP anyway. tp=1 and fat head shards never fire.
"""

import pytest

from models.demos.deepseek_v3_d_p.tt.mla import ttMLA


class _MlaStub:
    """Minimal stand-in exposing only the two attributes the predicate reads, so the *real*
    `ttMLA._needs_head_to_seq_reshard` getter is exercised without constructing a device model."""

    def __init__(self, num_heads: int, tp_factor: int):
        self.num_heads = num_heads
        self.tp_factor = tp_factor


def _needs_reshard(num_heads: int, tp_factor: int) -> bool:
    return ttMLA._needs_head_to_seq_reshard.fget(_MlaStub(num_heads, tp_factor))


@pytest.mark.parametrize(
    "num_heads, tp_factor, expected_skip, note",
    [
        # Thin head shard -> skip fires (the models this PR targets).
        (64, 4, True, "GLM-5.1/5.2 & DeepSeek-V4: 64h/tp4 -> 16 < 32"),
        (64, 8, True, "GLM at tp=8: 64h/tp8 -> 8 < 32"),
        (96, 4, True, "96h/tp4 -> 24 < 32"),
        # H/tp >= 32 but not a multiple of 32 -> still too thin for sparse_sdpa -> skip fires.
        (160, 4, True, "160h/tp4 -> 40, 40 % 32 != 0"),
        # Fat head shard -> skip does NOT fire; the gathered S/sp contract is genuinely needed.
        (128, 4, False, "DeepSeek-V3.2: 128h/tp4 -> 32 (exactly, divisible)"),
        (128, 2, False, "128h/tp2 -> 64"),
        (64, 2, False, "64h/tp2 -> 32 (exactly, divisible)"),
        # tp=1: no TP axis to gather/re-split -> never fires, regardless of head count.
        (64, 1, False, "tp=1: no tensor-parallel axis"),
        (16, 1, False, "tp=1 with a thin model"),
    ],
)
def test_needs_head_to_seq_reshard_truth_table(num_heads, tp_factor, expected_skip, note):
    assert _needs_reshard(num_heads, tp_factor) is expected_skip, note


def test_skip_boundary_at_32_heads_per_chip():
    """The threshold is exactly 32 heads/chip: 31 fires, 32 does not (guards an off-by-one that
    would either disable the skip for GLM or wrongly enable it for a fat-head model)."""
    assert _needs_reshard(124, 4) is True  # 124h/tp4 -> 31 per chip -> too thin -> fires
    assert _needs_reshard(128, 4) is False  # 128h/tp4 -> 32 per chip -> fat enough -> no skip
