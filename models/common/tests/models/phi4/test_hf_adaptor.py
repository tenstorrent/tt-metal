# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from transformers import Phi3Config, Phi3ForCausalLM

from models.common.models.phi4 import hf_adaptor
from models.common.models.phi4 import model as phi4_model
from models.common.models.phi4.hf_adaptor import (
    DEFAULT_HF_REVISION,
    Phi4ForCausalLM,
    Phi4RuntimeConfig,
    convert_hf_model_weights,
)


def _tiny_config(**overrides):
    values = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=128,
        original_max_position_embeddings=128,
        rope_theta=250_000.0,
        partial_rotary_factor=1.0,
        attention_bias=False,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        rope_scaling={"rope_type": "default", "rope_theta": 250_000.0, "partial_rotary_factor": 1.0},
    )
    values.update(overrides)
    return Phi3Config(**values)


def test_pinned_revision_and_runtime_cap_are_preserved(expect_error):
    assert DEFAULT_HF_REVISION == "187ef0342fff0eb3333be9f00389385e95ef0b61"
    runtime = Phi4RuntimeConfig(
        model_name="phi-4",
        model_cache_path=None,
        max_prefill_chunk_size=2048,
        max_context_len=16384,
        max_seq_len=4096,
        trace_prefill_supported_seq_lens=(128, 1024),
    )
    assert runtime.max_prefill_batch_size == 8
    assert runtime.can_enable_trace(128)
    assert runtime.can_enable_trace(1024, num_cached_tokens=64)
    assert not runtime.can_enable_trace(2048)
    assert hf_adaptor._trace_seq_lens(2, 2048, 4096) == (128, 1024)
    with expect_error(ValueError, "TP2"):
        hf_adaptor._trace_seq_lens(8, 2048, 4096)


def test_product_binds_runtime_config_and_chatml_stop_token():
    model = SimpleNamespace(config=SimpleNamespace(max_seq_len=4096), model_args=None)
    runtime = Phi4RuntimeConfig(
        model_name="phi-4",
        model_cache_path=None,
        max_prefill_chunk_size=2048,
        max_context_len=16384,
        max_seq_len=4096,
        trace_prefill_supported_seq_lens=(128, 1024),
    )
    product = Phi4ForCausalLM(model=model, tokenizer=SimpleNamespace(stop_tokens=[100265]), runtime_config=runtime)
    assert model.model_args is runtime
    assert product.generation_config.stop_token_ids == (100265,)
    assert product.max_seq_len == 4096
    assert product.max_context_len == 16384


def test_tokenizer_threads_pin_and_preserves_chatml_end(monkeypatch):
    tokenizer = SimpleNamespace(
        eos_token_id=2,
        convert_tokens_to_ids=lambda token: {"<|im_end|>": 7, "<|im_start|>": 8}[token],
    )
    seen = {}

    def fake_from_pretrained(model, **kwargs):
        seen.update(model=model, **kwargs)
        return tokenizer

    monkeypatch.setattr(hf_adaptor.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    assert hf_adaptor.load_tokenizer("microsoft/phi-4") is tokenizer
    assert seen["revision"] == DEFAULT_HF_REVISION
    assert tokenizer.stop_tokens == [2, 7]


def test_encode_prompt_preserves_exact_phi4_chatml_request():
    calls = []
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: calls.append((messages, kwargs))
        or {"input_ids": [[101, 102, 103]]},
    )

    assert hf_adaptor.encode_prompt(tokenizer, "Hello", "Be concise") == [101, 102, 103]
    assert calls == [
        (
            [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Hello"},
            ],
            {"add_generation_prompt": True, "tokenize": True},
        )
    ]


def test_checkpoint_contract_requires_full_plain_theta_250k_rope(expect_error):
    config = _tiny_config()
    hf_adaptor._validate_checkpoint_config(config)
    assert config.rope_parameters["rope_theta"] == 250_000.0
    assert config.rope_parameters["partial_rotary_factor"] == 1.0
    with expect_error(ValueError, "full-head RoPE"):
        hf_adaptor._validate_checkpoint_config(_tiny_config(partial_rotary_factor=0.5))
    with expect_error(ValueError, "theta=250,000"):
        hf_adaptor._validate_checkpoint_config(
            _tiny_config(rope_scaling={"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 1.0})
        )


def test_conversion_splits_fused_qkv_and_gate_up_and_keeps_untied_head():
    config = _tiny_config()
    hf = Phi3ForCausalLM(config).eval()
    weights = convert_hf_model_weights(
        hf,
        config,
        n_layers=1,
        num_devices=2,
        rope_table_len=128,
        head_dim=16,
    )
    layer = weights.layers[0]
    assert layer.wqkv.shape == (1, 1, 64, 128)
    assert layer.wo.shape == (1, 1, 64, 64)
    assert layer.w1.shape == layer.w3.shape == (64, 128)
    assert layer.w2.shape == (128, 64)
    torch.testing.assert_close(weights.lm_head, hf.lm_head.weight.detach().to(torch.bfloat16))
    assert weights.lm_head.data_ptr() != weights.embedding.data_ptr()


def test_config_builder_is_owned_by_model_module():
    assert hf_adaptor.build_phi4_transformer_config is phi4_model.build_phi4_transformer_config
    assert phi4_model.build_phi4_transformer_config.__module__ == phi4_model.__name__


def test_post_attention_norm_program_and_memory_use_same_mlp_grid(monkeypatch):
    grid = SimpleNamespace(num_cores=32)
    program = object()
    memory = object()
    captured = {}

    monkeypatch.setattr(phi4_model, "get_padded_hidden_dim", lambda *_: 17920)
    monkeypatch.setattr(phi4_model, "_dram_shard_core_grid_k_n", lambda *_: grid)
    monkeypatch.setattr(
        phi4_model,
        "_create_sharded_norm_program_config",
        lambda dim, selected_grid, rows, tile: captured.update(program=(dim, selected_grid, rows, tile)) or program,
    )
    monkeypatch.setattr(
        phi4_model.ttnn,
        "create_sharded_memory_config",
        lambda shape, selected_grid, *args, **kwargs: captured.update(memory=(shape, selected_grid)) or memory,
    )

    assert phi4_model._post_attn_norm_decode_configs(
        dim=5120,
        hidden_dim=17920,
        num_devices=2,
        max_batch_size=32,
    ) == (program, memory)
    assert captured["program"] == (5120, grid, 32, 32)
    assert captured["memory"] == ((32, 160), grid)


def test_decoder_prefill_calls_attention_prefill_surface(monkeypatch):
    calls = []
    norm = SimpleNamespace(prefill_forward=lambda x: x)
    attention = SimpleNamespace(
        prefill_forward=lambda x, rot, **kwargs: calls.append((x, rot, kwargs)) or "attention-output"
    )
    mlp = SimpleNamespace(prefill_forward=lambda x: "mlp-output")
    layer = phi4_model.Phi4DecoderLayer(
        input_layernorm=norm,
        attention=attention,
        post_attention_layernorm=norm,
        mlp=mlp,
    )
    monkeypatch.setattr(phi4_model, "_all_gather_rmsnorm_tensor", lambda _norm, x, **_: x)
    monkeypatch.setattr(phi4_model.ttnn, "add", lambda left, right, **_: f"{left}+{right}")

    result = layer.prefill_forward(
        "hidden",
        "rotary",
        chunk_start_idx=32,
        chunk_start_idx_tensor="chunk-index",
    )
    assert result == "hidden+attention-output+mlp-output"
    assert calls[0][2]["chunk_start_idx"] == 32
    assert calls[0][2]["chunk_start_idx_tensor"] == "chunk-index"
