# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU check that LTX_FUSE_GATE_ON_DEVICE builds the weights the LTX_FUSE_GATE cache holds.

fold_gate_on_device concatenates, per TP device, the unfused Q/QKV shard with the gate shard
zero-padded to a tile. This test runs the real _prepare_torch_state for both layouts and emulates
ColParallelLinear's per-device sharding, so it needs no device.
"""

from types import SimpleNamespace

import pytest
import torch

from models.tt_dit.models.transformers.ltx.attention_ltx import LTXAttention

TILE = 32


def _attn(num_heads, head_dim, tp, is_self, fuse_gate):
    n_local = num_heads // tp
    return SimpleNamespace(
        num_heads=num_heads,
        head_dim=head_dim,
        n_local_heads=n_local,
        gate_padded_per_device=-(-n_local // TILE) * TILE,
        parallel_config=SimpleNamespace(tensor_parallel=SimpleNamespace(factor=tp)),
        fuse_gate=fuse_gate,
        is_self=is_self,
    )


def _checkpoint(num_heads, head_dim, q_in, kv_in, is_self, gen):
    dim = num_heads * head_dim

    def rand(*shape):
        return torch.randn(*shape, generator=gen).to(torch.bfloat16)

    state = {"to_q.weight": rand(dim, q_in), "to_q.bias": rand(dim)}
    for name in ("to_k", "to_v"):
        state[f"{name}.weight"] = rand(dim, q_in if is_self else kv_in)
        state[f"{name}.bias"] = rand(dim)
    state["to_gate_logits.weight"] = rand(num_heads, q_in)
    state["to_gate_logits.bias"] = rand(num_heads)
    return state


def _prepared(attn, checkpoint):
    state = {k: v.clone() for k, v in checkpoint.items()}
    LTXAttention._prepare_torch_state(attn, state)
    return state


def _shards(weight_or_bias, tp, is_bias):
    """ColParallelLinear stores weight.T / bias as (1, N), split on N across the TP axis."""
    t = weight_or_bias.reshape(1, -1) if is_bias else weight_or_bias.transpose(0, 1)
    return torch.chunk(t, tp, dim=1)


def _fold(proj_shard, gate_shard, padded):
    pad = torch.zeros(proj_shard.shape[0], padded - gate_shard.shape[1], dtype=gate_shard.dtype)
    return torch.cat([proj_shard, gate_shard, pad], dim=1)


# (num_heads, head_dim, q_in, kv_in, is_self): video self/text-cross, audio self, A->V, V->A.
SHAPES = [
    (32, 128, 4096, 4096, True),
    (32, 128, 4096, 4096, False),
    (32, 64, 2048, 2048, True),
    (32, 64, 4096, 2048, False),
    (32, 64, 2048, 4096, False),
]


@pytest.mark.parametrize("tp", [4, 2])
@pytest.mark.parametrize("num_heads,head_dim,q_in,kv_in,is_self", SHAPES)
def test_fold_matches_fused_cache_layout(num_heads, head_dim, q_in, kv_in, is_self, tp):
    gen = torch.Generator().manual_seed(0)
    checkpoint = _checkpoint(num_heads, head_dim, q_in, kv_in, is_self, gen)
    fused = _prepared(_attn(num_heads, head_dim, tp, is_self, fuse_gate=True), checkpoint)
    unfused_attn = _attn(num_heads, head_dim, tp, is_self, fuse_gate=False)
    unfused = _prepared(unfused_attn, checkpoint)
    proj = "to_qkv" if is_self else "to_q"
    padded = unfused_attn.gate_padded_per_device

    for kind in ("weight", "bias"):
        is_bias = kind == "bias"
        fused_shards = _shards(fused[f"{proj}.{kind}"], tp, is_bias)
        proj_shards = _shards(unfused[f"{proj}.{kind}"], tp, is_bias)
        gate_shards = _shards(unfused[f"to_gate_logits.{kind}"], tp, is_bias)
        for d in range(tp):
            folded = _fold(proj_shards[d], gate_shards[d], padded)
            assert torch.equal(folded, fused_shards[d]), f"{proj}.{kind} device {d}"

            # The check must be able to fail: the gate placed first is a different layout.
            mutant = torch.cat([folded[:, proj_shards[d].shape[1] :], proj_shards[d]], dim=1)
            assert not torch.equal(mutant, fused_shards[d])
