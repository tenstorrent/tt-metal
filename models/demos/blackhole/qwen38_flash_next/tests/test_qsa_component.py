# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.tt.qsa import Qwen38QSA, Qwen38QSAState, Qwen38QSAWeights

CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))
TRANSFORMERS_SRC = os.environ.get("QWEN38_TRANSFORMERS_SRC")


@pytest.fixture(scope="module")
def weights():
    return Qwen38QSAWeights.from_checkpoint(Qwen38Checkpoint(CHECKPOINT), layer_idx=3)


def _position_embeddings(sequence: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(sequence, dtype=torch.float32).view(1, sequence, 1)
    frequencies = torch.linspace(0.001, 0.037, 32).view(1, 1, 32)
    angles = positions * frequencies
    return torch.cat((angles.cos(), angles.cos()), dim=-1).to(torch.bfloat16), torch.cat(
        (angles.sin(), angles.sin()), dim=-1
    ).to(torch.bfloat16)


def _causal_additive_mask(sequence: int) -> torch.Tensor:
    visible = torch.ones(sequence, sequence, dtype=torch.bool).tril()
    return torch.where(visible, torch.tensor(0.0), torch.tensor(torch.finfo(torch.float32).min)).view(
        1, 1, sequence, sequence
    )


def test_exact_qsa_tp4_shards_and_two_kv_groups(weights):
    assert weights.dimensions.query_heads == 24
    assert weights.dimensions.kv_heads == 2
    assert weights.dimensions.query_heads_per_device == 6
    assert weights.dimensions.index_query_heads_per_device == 1

    shards = [weights.device_shard(index) for index in range(4)]
    assert [shard.kv_head_index for shard in shards] == [0, 0, 1, 1]
    assert all(shard.qg.shape == (3072, 2560) for shard in shards)
    assert all(shard.k.shape == (256, 2560) for shard in shards)
    assert all(shard.v.shape == (256, 2560) for shard in shards)
    assert all(shard.out.shape == (2560, 1536) for shard in shards)
    assert all(shard.index_q.shape == (128, 2560) for shard in shards)
    assert all(shard.index_k.shape == (128, 2560) for shard in shards)

    query, gate = zip(*(shard.split_query_gate() for shard in shards))
    full_query, full_gate = weights.split_query_gate()
    torch.testing.assert_close(torch.cat(query), full_query, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(gate), full_gate, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([shard.out for shard in shards], dim=1), weights.out, rtol=0, atol=0)
    torch.testing.assert_close(shards[0].k, shards[1].k, rtol=0, atol=0)
    torch.testing.assert_close(shards[2].k, shards[3].k, rtol=0, atol=0)
    assert not torch.equal(shards[0].k, shards[2].k)
    torch.testing.assert_close(shards[0].index_k, shards[3].index_k, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([shard.index_q for shard in shards]), weights.index_qk[:512], rtol=0, atol=0)


@pytest.mark.skipif(not TRANSFORMERS_SRC, reason="set QWEN38_TRANSFORMERS_SRC to pinned Transformers source")
def test_exact_checkpoint_qsa_matches_pinned_transformers(weights):
    sys.path.insert(0, str(Path(TRANSFORMERS_SRC)))
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention

    config = Qwen4ExpTextConfig.from_pretrained(CHECKPOINT)
    config._attn_implementation = "eager"
    oracle = Qwen4ExpTextAttention(config, layer_idx=3).to(torch.bfloat16).eval()
    oracle.load_state_dict(weights.transformers_state_dict(), strict=True)

    torch.manual_seed(227)
    sequence = 8
    hidden = torch.randn(1, sequence, 2560, dtype=torch.bfloat16)
    position_embeddings = _position_embeddings(sequence)
    attention_mask = _causal_additive_mask(sequence)
    with torch.no_grad():
        expected, _ = oracle(hidden, position_embeddings, attention_mask, past_key_values=None)
        got, state, selected_mask = Qwen38QSA(weights).forward(hidden, position_embeddings, attention_mask)

    torch.testing.assert_close(got, expected, rtol=0.02, atol=0.02)
    # At S=8 the official 2,048-token budget covers every causal token.  This
    # proves the dense short-context oracle without claiming final long-QSA perf.
    torch.testing.assert_close(selected_mask, attention_mask == 0)
    assert state.length == sequence


class _ReferenceIndexedCache:
    def __init__(self):
        self.raw_keys = None
        self.keys = None
        self.values = None

    def update_indexer(self, raw_keys, layer_idx):
        del layer_idx
        self.raw_keys = raw_keys if self.raw_keys is None else torch.cat((self.raw_keys, raw_keys), dim=1)
        return self.raw_keys

    def update(self, keys, values, layer_idx):
        del layer_idx
        self.keys = keys if self.keys is None else torch.cat((self.keys, keys), dim=-2)
        self.values = values if self.values is None else torch.cat((self.values, values), dim=-2)
        return self.keys, self.values


@pytest.mark.skipif(not TRANSFORMERS_SRC, reason="set QWEN38_TRANSFORMERS_SRC to pinned Transformers source")
def test_qsa_prefill_equals_tokenwise_cache_and_matches_pinned_transformers(weights):
    sys.path.insert(0, str(Path(TRANSFORMERS_SRC)))
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention

    config = Qwen4ExpTextConfig.from_pretrained(CHECKPOINT)
    config._attn_implementation = "eager"
    oracle = Qwen4ExpTextAttention(config, layer_idx=3).to(torch.bfloat16).eval()
    oracle.load_state_dict(weights.transformers_state_dict(), strict=True)

    generator = torch.Generator().manual_seed(229)
    sequence = 6
    hidden = torch.randn((1, sequence, 2560), generator=generator, dtype=torch.bfloat16)
    full_positions = _position_embeddings(sequence)
    full_mask = _causal_additive_mask(sequence)
    qsa = Qwen38QSA(weights)
    with torch.no_grad():
        full_output, full_state, _ = qsa.forward(hidden, full_positions, full_mask)
        state = None
        reference_cache = _ReferenceIndexedCache()
        pieces = []
        reference_pieces = []
        for position in range(sequence):
            positions = tuple(value[:, : position + 1] for value in full_positions)
            mask = torch.zeros((1, 1, 1, position + 1), dtype=torch.float32)
            output, state, selected = qsa.forward(hidden[:, position : position + 1], positions, mask, state=state)
            reference_output, _ = oracle(
                hidden[:, position : position + 1], positions, mask, past_key_values=reference_cache
            )
            pieces.append(output)
            reference_pieces.append(reference_output)
            assert selected.shape == (1, 1, 1, position + 1)

    tokenwise = torch.cat(pieces, dim=1)
    reference_tokenwise = torch.cat(reference_pieces, dim=1)
    torch.testing.assert_close(tokenwise, full_output, rtol=0.02, atol=0.02)
    torch.testing.assert_close(tokenwise, reference_tokenwise, rtol=0.02, atol=0.02)
    torch.testing.assert_close(state.raw_index_keys, reference_cache.raw_keys, rtol=0.02, atol=0.02)
    torch.testing.assert_close(state.keys, reference_cache.keys, rtol=0.02, atol=0.02)
    torch.testing.assert_close(state.values, reference_cache.values, rtol=0.02, atol=0.02)
    torch.testing.assert_close(state.raw_index_keys, full_state.raw_index_keys, rtol=0.02, atol=0.02)


def test_qsa_state_append_is_immutable_and_prefix_rolls_back(weights, expect_error):
    qsa = Qwen38QSA(weights)
    hidden = torch.zeros((1, 2, 2560), dtype=torch.bfloat16)
    _, state, _ = qsa.forward(hidden, _position_embeddings(2), _causal_additive_mask(2))
    _, appended, _ = qsa.forward(
        hidden[:, :1],
        _position_embeddings(3),
        torch.zeros((1, 1, 1, 3), dtype=torch.float32),
        state=state,
    )

    assert isinstance(state, Qwen38QSAState)
    assert state.length == 2
    assert appended.length == 3
    rolled_back = appended.prefix(2)
    assert torch.equal(rolled_back.raw_index_keys, state.raw_index_keys)
    assert torch.equal(rolled_back.keys, state.keys)
    assert torch.equal(rolled_back.values, state.values)
    with expect_error(ValueError, "outside cached QSA length"):
        appended.prefix(4)
