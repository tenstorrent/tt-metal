# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every K block size and DRAM core count must divide its projection's K, on both platforms.

These matmuls take their K blocking from the policy and reject -- or silently mis-compute -- a
value that does not divide K in tiles. K is per-device for the roles that shard it, so halving
the tensor-parallel width moves four of these six numbers.
"""

import pytest

from models.demos.qwen38_27b_qb2.tt.decoder import DEFAULT_POLICY
from models.demos.qwen38_27b_qb2.tt.decoder_tp import _TP_POLICY, measured_policy

# Qwen3.8-27B: hidden 5120, 24 query heads of 256, intermediate 17408.
_HIDDEN, _Q_HEADS, _HEAD_DIM, _INTERMEDIATE = 5120, 24, 256, 17408
_KINDS = ["linear_attention", "full_attention"]


def _k_tiles(tp):
    """Per-role K in tiles, as the on-device audit reports it."""
    return {
        "attention": _HIDDEN // 32,
        "gate": _HIDDEN // 32,
        "up": _HIDDEN // 32,
        "output": (_Q_HEADS * _HEAD_DIM // tp) // 32,
        "down": (_INTERMEDIATE // tp) // 32,
    }


def _policy(kind, tp):
    policy = dict(DEFAULT_POLICY)
    policy.update(measured_policy(kind))
    policy.update(_TP_POLICY.get(tp, {}))
    return policy


@pytest.mark.parametrize("tp", [4, 8], ids=["tp4", "tp8"])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("role", sorted(_k_tiles(4)))
def test_prefill_1d_block_divides_k(kind, tp, role):
    policy = _policy(kind, tp)
    block = policy.get("prefill_1d_" + role + "_k", policy["prefill_1d_k"])
    kt = _k_tiles(tp)[role]
    assert kt % block == 0, f"{role} Kt={kt} is not divisible by in0_block_w={block} at TP={tp}"


@pytest.mark.parametrize("tp", [4, 8], ids=["tp4", "tp8"])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("role", sorted(_k_tiles(4)))
def test_dram_shard_divides_k_and_fits_the_grid(kind, tp, role):
    policy = _policy(kind, tp)
    cores = policy[role + "_cores"]
    kt = _k_tiles(tp)[role]
    # DRAM-sharded matmul derives per_core_K from the in0 shard width and requires it to
    # divide K exactly, so an inexact core count is rejected outright.
    assert kt % cores == 0, f"{role} Kt={kt} does not shard over {cores} cores at TP={tp}"
    # 64 workers on a T3K, about 110 on QB2.
    assert cores <= (64 if tp == 8 else 110)


@pytest.mark.parametrize("kind", _KINDS)
def test_wormhole_takes_one_dram_reader_per_bank(kind):
    # num_workers_per_dram_bank > 1 is Blackhole-only; the projections and head must agree.
    policy = _policy(kind, 8)
    assert all(policy[role + "_readers"] == 1 for role in _k_tiles(8))
    assert _TP_POLICY[8]["head_readers"] == 1


@pytest.mark.parametrize("kind", _KINDS)
def test_qb2_keeps_its_measured_blocking(kind):
    policy = _policy(kind, 4)
    assert policy["down_cores"] == 8 and policy["down_block"] == 17
    assert policy["attention_readers"] == 2 and policy["gate_readers"] == 3
