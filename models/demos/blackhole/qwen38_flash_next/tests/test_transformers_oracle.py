# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.reference import (
    build_ngram_hash_spec,
    causal_depthwise_conv1d,
    gated_delta_recurrent,
    gated_residual_read,
    gated_residual_write,
    ngram_token_ids,
    qsa_selected_token_mask,
)

TRANSFORMERS_SRC = os.environ.get("QWEN38_TRANSFORMERS_SRC")
if not TRANSFORMERS_SRC:
    pytest.skip("set QWEN38_TRANSFORMERS_SRC to the pinned Transformers src directory", allow_module_level=True)
sys.path.insert(0, str(Path(TRANSFORMERS_SRC)))

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import (  # noqa: E402
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextNGramEmbedding,
    Qwen4ExpTextQSAIndexer,
    causal_conv1d_update,
    torch_recurrent_gated_delta_rule,
)


def _small_config(**overrides):
    values = dict(
        vocab_size=97,
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts_per_tok=2,
        num_experts=4,
        layer_types=["full_attention"],
        hc_count=4,
        hc_lowrank=3,
        eos_token_id=96,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=4,
        indexer_compress_ratio=2,
        rope_parameters={"partial_rotary_factor": 1.0, "rope_theta": 10_000.0},
    )
    values.update(overrides)
    return Qwen4ExpTextConfig(**values)


def test_gr_matches_pinned_transformers_module():
    torch.manual_seed(101)
    config = _small_config()
    module = Qwen4ExpTextGatedResidual(config)
    residual = torch.randn(2, config.hc_count * config.hidden_size)
    block = torch.randn(2, config.hidden_size)

    hf_read, hf_residual, hf_coeff = module(residual)
    read, normed = gated_residual_read(
        residual,
        module.hc_norm.weight,
        module.input_mix_weight_down.weight,
        module.input_mix_weight_up.weight,
        config.hc_count,
        config.hidden_size,
        config.rms_norm_eps,
    )
    written = gated_residual_write(
        block,
        residual,
        normed,
        module.block_inject_weight.weight,
        config.hc_count,
        config.hidden_size,
    )
    hf_written = (
        hf_residual.unflatten(-1, (config.hc_count, config.hidden_size)) + hf_coeff.unsqueeze(-1) * block.unsqueeze(-2)
    ).flatten(-2)
    torch.testing.assert_close(read, hf_read)
    torch.testing.assert_close(written, hf_written)


def test_ngram_ids_match_pinned_transformers_including_eos_boundaries():
    config = _small_config(
        layer_types=["linear_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=8,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=101,
        make_ngram_vocab_size_divisible_by=8,
        seed=1234,
    )
    module = Qwen4ExpTextNGramEmbedding(config, embedding_dim=8, layer_idx=0)
    captured = []
    hook = module.ngram_embedding.register_forward_pre_hook(lambda _module, args: captured.append(args[0].clone()))
    tokens = torch.tensor([[1, 2, 3, 96, 4, 5], [9, 96, 10, 11, 12, 13]])
    module(tokens, past_key_values=None)
    hook.remove()

    spec = build_ngram_hash_spec(97, 3, 2, 101, 0, 1234, 8)
    got, _ = ngram_token_ids(tokens, None, 96, spec)
    torch.testing.assert_close(got, captured[0])
    torch.testing.assert_close(spec.layer_multipliers, module.layer_multipliers)
    torch.testing.assert_close(spec.head_vocab_sizes, module.ngram_heads_vocab_sizes)
    torch.testing.assert_close(spec.head_offsets, module.ngram_heads_offsets)


def test_gdn_recurrent_rule_matches_pinned_transformers_fallback():
    torch.manual_seed(103)
    q = torch.randn(2, 5, 3, 4, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(2, 5, 3, 6, dtype=torch.bfloat16)
    g = -torch.rand(2, 5, 3)
    beta = torch.sigmoid(torch.randn(2, 5, 3))
    state = torch.randn(2, 3, 4, 6)

    expected_out, expected_state = torch_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    got_out, got_state = gated_delta_recurrent(q, k, v, g, beta, state)
    torch.testing.assert_close(got_out, expected_out)
    torch.testing.assert_close(got_state, expected_state)


def test_gdn_cached_conv_update_matches_pinned_transformers_fallback():
    torch.manual_seed(107)
    x = torch.randn(2, 3, 1)
    state = torch.randn(2, 3, 4)
    weight = torch.randn(3, 4)
    expected_state = state.clone()
    expected = causal_conv1d_update(x, expected_state, weight, activation="silu")
    got, got_state = causal_depthwise_conv1d(x, weight, initial_state=state, activation="silu")
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(got_state, expected_state)


def test_qsa_selection_matches_pinned_transformers_indexer():
    torch.manual_seed(109)
    config = _small_config()
    module = Qwen4ExpTextQSAIndexer(config, layer_idx=0)
    with torch.no_grad():
        module.index_qk_proj.weight.copy_(torch.eye(config.hidden_size))
        module.q_layernorm.weight.zero_()
        module.k_layernorm.weight.zero_()

    hidden = torch.randn(1, 7, config.hidden_size)
    cos = torch.ones(1, 7, config.indexer_head_dim)
    sin = torch.zeros_like(cos)
    visible = torch.ones(1, 1, 7, 7, dtype=torch.bool).tril()
    expected = module(hidden, (cos, sin), visible, past_key_values=None)

    q = hidden[..., : config.indexer_n_heads * config.indexer_head_dim]
    q = q.reshape(1, 7, config.indexer_n_heads, config.indexer_head_dim)
    raw_keys = hidden[..., -config.indexer_head_dim :]
    got = qsa_selected_token_mask(
        q,
        raw_keys,
        cos,
        sin,
        visible,
        q_norm_weight=module.q_layernorm.weight,
        k_norm_weight=module.k_layernorm.weight,
        token_budget=config.indexer_budget,
        compress_ratio=config.indexer_compress_ratio,
        eps=config.rms_norm_eps,
    )
    torch.testing.assert_close(got, expected)
