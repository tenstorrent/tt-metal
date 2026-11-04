# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Q/K-norm gamma tests for ``TransformerBlock.update_weights`` on Qwen3 (real weights).

Zero gammas must change the tokens; the read-back gammas must bring them back, in the same buffers.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from _ttt_sampler_utils import as_update_input, generate_one, open_sampler, to_torch_2d

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "Explain a tensor in a paragraph."
MAX_NEW_TOKENS = 32


@pytest.fixture(scope="module")
def sampler():
    with open_sampler(dummy_weights=False, model_source=MODEL_ID) as s:
        yield s


def _gamma(norm):
    return ttnn.to_torch(norm.weight).reshape(-1)


def _layer_hf(layer):
    """Read every weight of a decoder layer back as HF-shape torch tensors, keyed as ``update_weights`` takes them."""
    a, mlp = layer.attention, layer.feed_forward
    n_q, n_kv = a.n_heads * a.head_dim, a.n_kv_heads * a.head_dim
    q, k, v = torch.split(to_torch_2d(a.wqkv).transpose(0, 1), [n_q, n_kv, n_kv], dim=0)
    return {
        "self_attn.q_proj.weight": q,
        "self_attn.k_proj.weight": k,
        "self_attn.v_proj.weight": v,
        "self_attn.o_proj.weight": to_torch_2d(a.wo).transpose(0, 1),
        "self_attn.q_norm.weight": _gamma(a._q_norm_module),
        "self_attn.k_norm.weight": _gamma(a._k_norm_module),
        "mlp.gate_proj.weight": to_torch_2d(mlp.w1).transpose(0, 1),
        "mlp.up_proj.weight": to_torch_2d(mlp.w3).transpose(0, 1),
        "mlp.down_proj.weight": to_torch_2d(mlp.w2).transpose(0, 1),
        "input_layernorm.weight": _gamma(layer.attention_norm.norm),
        "post_attention_layernorm.weight": _gamma(layer.ff_norm.norm),
    }


def _push(layer, weights):
    layer.update_weights({key: as_update_input(w, layer.attention.mesh_device) for key, w in weights.items()})


def _qk_norm_addresses(layers):
    return [
        (layer.attention._q_norm_module.weight.buffer_address(), layer.attention._k_norm_module.weight.buffer_address())
        for layer in layers
    ]


def test_qk_norm_update_round_trip(sampler):
    layers = sampler.models[0].layers
    prompt_ids = sampler.tokenizer.encode(PROMPT, add_special_tokens=True)
    tokens = generate_one(sampler, prompt_ids, max_new_tokens=MAX_NEW_TOKENS)
    weights = [_layer_hf(layer) for layer in layers]
    addresses = _qk_norm_addresses(layers)

    for layer, w in zip(layers, weights):
        zeros = torch.zeros(layer.attention.head_dim)
        _push(layer, {**w, "self_attn.q_norm.weight": zeros, "self_attn.k_norm.weight": zeros})
    zeroed_tokens = generate_one(sampler, prompt_ids, max_new_tokens=MAX_NEW_TOKENS)
    assert zeroed_tokens != tokens, "zero gammas did not change the tokens"

    for layer, w in zip(layers, weights):
        _push(layer, w)
    assert generate_one(sampler, prompt_ids, max_new_tokens=MAX_NEW_TOKENS) == tokens
    assert _qk_norm_addresses(layers) == addresses


def test_layer_update_requires_qk_norm(sampler, expect_error):
    """A Qwen3 layer update with no Q/K-norm gammas raises ``KeyError`` before any weight changes."""
    projections_only = {f"self_attn.{p}_proj.weight": None for p in "qkvo"}
    with expect_error(KeyError, "self_attn.q_norm.weight"):
        sampler.models[0].layers[0].update_weights(projections_only)
