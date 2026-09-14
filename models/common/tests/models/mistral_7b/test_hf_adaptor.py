# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from transformers import MistralConfig, MistralForCausalLM

from models.common.models.mistral_7b import hf_adaptor
from models.common.models.mistral_7b import model as mistral_model
from models.common.models.mistral_7b import weight_utils
from models.common.models.mistral_7b.hf_adaptor import (
    Mistral7BForCausalLM,
    Mistral7BRuntimeConfig,
    _trace_seq_lens,
    convert_hf_model_weights,
)


def test_runtime_config_preserves_per_sku_trace_and_batched_prefill_policy():
    runtime = Mistral7BRuntimeConfig(
        model_name="Mistral-7B-Instruct-v0.3",
        model_cache_path=None,
        max_prefill_chunk_size=2048,
        max_context_len=32768,
        max_seq_len=4096,
        trace_prefill_supported_seq_lens=(128,),
        max_prefill_batch_size=8,
    )
    assert runtime.can_enable_trace(128, num_cached_tokens=32)
    assert not runtime.can_enable_trace(1024)
    assert runtime.supports_batched_prefill
    assert runtime.max_prefill_batch_size == 8
    assert _trace_seq_lens(1, 2048, 4096) == (128,)
    assert _trace_seq_lens(2, 2048, 4096) == (128, 1024)
    assert _trace_seq_lens(8, 2048, 4096) == (128, 1024)


def test_product_binds_runtime_config_and_eos_stop_token():
    model = SimpleNamespace(config=SimpleNamespace(max_seq_len=4096), model_args=None)
    tokenizer = SimpleNamespace(stop_tokens=[2])
    runtime = Mistral7BRuntimeConfig(
        model_name="model",
        model_cache_path=None,
        max_prefill_chunk_size=2048,
        max_context_len=32768,
        max_seq_len=4096,
        trace_prefill_supported_seq_lens=(128, 1024),
    )
    product = Mistral7BForCausalLM(model=model, tokenizer=tokenizer, runtime_config=runtime)
    assert model.model_args is runtime
    assert product.generation_config.stop_token_ids == (2,)
    assert product.max_seq_len == 4096
    assert product.max_context_len == 32768


def test_tokenizer_adds_only_eos_and_threads_optional_revision(monkeypatch):
    tokenizer = SimpleNamespace(eos_token_id=2)
    seen = {}

    def fake_from_pretrained(model, **kwargs):
        seen.update(model=model, **kwargs)
        return tokenizer

    monkeypatch.setattr(hf_adaptor.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    assert hf_adaptor.load_tokenizer("mistralai/Mistral-7B-Instruct-v0.3", "revision") is tokenizer
    assert tokenizer.stop_tokens == [2]
    assert seen["revision"] == "revision"


def test_checkpoint_contract_preserves_plain_rope_and_full_attention():
    config = MistralConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_theta=1_000_000.0,
        sliding_window=None,
        attention_bias=False,
    )
    hf_adaptor._validate_checkpoint_config(config)
    assert config.rope_parameters["rope_theta"] == 1_000_000.0
    assert config.sliding_window is None
    assert config.attention_bias is False


def test_hf_rope_tables_are_derived_from_the_checkpoint_rotary_module():
    config = MistralConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=1_000_000.0,
        sliding_window=None,
    )
    hf = MistralForCausalLM(config).eval()
    table_len = 128
    head_dim = 16
    cos, sin = weight_utils.build_rope_cos_sin_torch(hf.model.rotary_emb, table_len, head_dim, torch.bfloat16)
    x = torch.zeros(1, 1, table_len, head_dim, dtype=torch.bfloat16)
    positions = torch.arange(table_len).unsqueeze(0)
    with torch.no_grad():
        hf_cos, hf_sin = hf.model.rotary_emb(x, positions)
    expected_cos, expected_sin = weight_utils.permute_hf_rope_to_meta_tables(hf_cos.float(), hf_sin.float())
    torch.testing.assert_close(cos, expected_cos.to(torch.bfloat16))
    torch.testing.assert_close(sin, expected_sin.to(torch.bfloat16))


def test_conversion_preserves_biasless_attention_and_untied_lm_head():
    config = MistralConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=128,
        rope_theta=1_000_000.0,
        sliding_window=None,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    hf = MistralForCausalLM(config).eval()
    weights = convert_hf_model_weights(hf, n_layers=1, num_devices=2, rope_table_len=128, head_dim=16)
    layer = weights.layers[0]
    assert layer.wqkv.shape == (1, 1, 64, 128)
    assert layer.wo.shape == (1, 1, 64, 64)
    assert layer.w1.shape == layer.w3.shape == (64, 128)
    assert layer.w2.shape == (128, 64)
    torch.testing.assert_close(weights.lm_head, hf.lm_head.weight.detach().to(torch.bfloat16))
    assert weights.lm_head.data_ptr() != weights.embedding.data_ptr()


def test_config_builder_is_owned_by_model_module():
    assert hf_adaptor.build_mistral_7b_transformer_config is mistral_model.build_mistral_7b_transformer_config
    assert mistral_model.build_mistral_7b_transformer_config.__module__ == mistral_model.__name__


def test_post_attention_norm_program_and_memory_use_same_mlp_grid(monkeypatch):
    grid = SimpleNamespace(num_cores=32)
    program = object()
    memory = object()
    captured = {}

    monkeypatch.setattr(mistral_model, "get_padded_hidden_dim", lambda *_: 14336)
    monkeypatch.setattr(mistral_model, "_dram_shard_core_grid_k_n", lambda *_: grid)
    monkeypatch.setattr(
        mistral_model,
        "_create_sharded_norm_program_config",
        lambda dim, selected_grid, rows, tile: captured.update(program=(dim, selected_grid, rows, tile)) or program,
    )
    monkeypatch.setattr(
        mistral_model.ttnn,
        "create_sharded_memory_config",
        lambda shape, selected_grid, *args, **kwargs: captured.update(memory=(shape, selected_grid)) or memory,
    )

    assert mistral_model._post_attn_norm_decode_configs(
        dim=4096,
        hidden_dim=14336,
        num_devices=8,
        max_batch_size=32,
    ) == (program, memory)
    assert captured["program"] == (4096, grid, 32, 32)
    assert captured["memory"] == ((32, 128), grid)
