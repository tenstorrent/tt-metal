# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.tt.ple import Qwen38HostPLEEmbedding, Qwen38PLE, Qwen38PLEWeights

CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))
TRANSFORMERS_SRC = os.environ.get("QWEN38_TRANSFORMERS_SRC")


@pytest.fixture(scope="module")
def checkpoint():
    return Qwen38Checkpoint(CHECKPOINT)


@pytest.fixture(scope="module")
def embedding(checkpoint):
    return Qwen38HostPLEEmbedding(checkpoint)


@pytest.fixture(scope="module")
def weights(checkpoint):
    return Qwen38PLEWeights.from_checkpoint(checkpoint)


def test_random_access_rows_match_safetensors_slice_without_materializing_table(checkpoint):
    name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    indices = torch.tensor([[0, 17], [2_500_010, 17]])
    got = checkpoint.tensor_rows(name, indices)
    expected = torch.stack(
        [checkpoint.tensor_slice(name, (int(index), slice(None))) for index in indices.flatten()]
    ).reshape(2, 2, 160)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_exact_host_ngram_lookup_is_split_invariant_and_shards_640_per_device(embedding):
    tokens = torch.tensor([[17, 29, 31, 248044, 43, 47]])
    full, full_context = embedding.lookup(tokens)
    first, first_context = embedding.lookup(tokens[:, :4])
    second, second_context = embedding.lookup(tokens[:, 4:], first_context)

    torch.testing.assert_close(torch.cat((first, second), dim=1), full, rtol=0, atol=0)
    torch.testing.assert_close(second_context, full_context, rtol=0, atol=0)
    assert full.shape == (1, 6, 2560)
    device_parts = embedding.shard_result(full)
    assert len(device_parts) == 4
    assert all(part.shape == (1, 6, 640) for part in device_parts)
    torch.testing.assert_close(torch.cat(device_parts, dim=-1), full, rtol=0, atol=0)


@pytest.mark.skipif(not TRANSFORMERS_SRC, reason="set QWEN38_TRANSFORMERS_SRC to pinned Transformers source")
def test_exact_ple_post_lookup_matches_pinned_transformers(embedding, weights):
    sys.path.insert(0, str(Path(TRANSFORMERS_SRC)))
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

    # Instantiate the official post-lookup path with a tiny temporary hash
    # vocabulary, then replace its embedding module.  This avoids allocating
    # the checkpoint's 51.2B host-resident n-gram parameters in the oracle.
    config = Qwen4ExpTextConfig.from_pretrained(CHECKPOINT)
    config.ngram_vocab_size_base = 101
    oracle = Qwen4ExpTextPLELayer(config, layer_idx=1, ple_layer_index=0).to(torch.bfloat16).eval()

    tokens = torch.tensor([[101, 202, 303]])
    exact_embeddings, _ = embedding.lookup(tokens)

    class FixedEmbedding(nn.Module):
        def forward(self, input_ids, past_key_values):
            assert torch.equal(input_ids, tokens)
            assert past_key_values is None
            return exact_embeddings

    oracle.ple_embedding = FixedEmbedding()
    oracle.load_state_dict(weights.transformers_state_dict(), strict=True)

    torch.manual_seed(229)
    hidden = torch.randn(1, 3, 10240, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = oracle(hidden, tokens, past_key_values=None)
        got, state = Qwen38PLE(embedding, weights).forward(hidden, tokens)

    torch.testing.assert_close(got, expected, rtol=0.02, atol=0.02)
    assert state.token_context.shape == (1, 2)
    assert state.conv.shape == (1, 10240, 9)


def test_exact_ple_prefill_equals_tokenwise_state_transition(embedding, weights):
    torch.manual_seed(233)
    tokens = torch.tensor([[501, 502, 503]])
    hidden = torch.randn(1, 3, 10240, dtype=torch.bfloat16)
    component = Qwen38PLE(embedding, weights)

    full_output, full_state = component.forward(hidden, tokens)
    state = None
    pieces = []
    for position in range(tokens.shape[1]):
        output, state = component.forward(hidden[:, position : position + 1], tokens[:, position : position + 1], state)
        pieces.append(output)

    torch.testing.assert_close(torch.cat(pieces, dim=1), full_output, rtol=0.02, atol=0.02)
    torch.testing.assert_close(state.token_context, full_state.token_context, rtol=0, atol=0)
    torch.testing.assert_close(state.conv, full_state.conv, rtol=0.01, atol=1e-3)
