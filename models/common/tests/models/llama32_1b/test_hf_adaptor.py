# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from models.common.models.llama32_1b import hf_adaptor
from models.common.models.llama32_1b import model as llama_model
from models.common.models.llama32_1b import weight_utils
from models.common.models.llama32_1b.hf_adaptor import (
    Llama32_1BForCausalLM,
    Llama32_1BRuntimeConfig,
    _trace_seq_lens,
    convert_hf_model_weights,
)

LLAMA32_ROPE_PARAMETERS = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_theta": 500000.0,
}


def test_runtime_config_preserves_trace_and_batched_prefill_policy():
    runtime = Llama32_1BRuntimeConfig(
        model_name="Llama-3.2-1B-Instruct",
        model_cache_path=None,
        max_prefill_chunk_size=2048,
        max_context_len=131072,
        max_seq_len=4096,
        trace_prefill_supported_seq_lens=(128, 1024),
    )
    assert runtime.can_enable_trace(128, num_cached_tokens=32)
    assert runtime.can_enable_trace(1024)
    assert not runtime.can_enable_trace(2048)
    assert runtime.supports_batched_prefill
    assert runtime.max_prefill_batch_size == 32
    assert runtime.batched_prefill_batched_extract


def test_trace_matrix_is_device_specific_and_bounded():
    assert _trace_seq_lens(1, 2048, 4096) == (128,)
    assert _trace_seq_lens(2, 2048, 4096) == (128, 1024)
    assert _trace_seq_lens(8, 2048, 4096) == (128, 1024)


def test_product_binds_runtime_config_unconditionally():
    model = SimpleNamespace(config=SimpleNamespace(max_seq_len=4096), model_args=None)
    tokenizer = SimpleNamespace(stop_tokens=[128001])
    runtime = Llama32_1BRuntimeConfig(
        model_name="model",
        model_cache_path=None,
        max_prefill_chunk_size=2048,
        max_context_len=131072,
        max_seq_len=4096,
        trace_prefill_supported_seq_lens=(128,),
    )
    product = Llama32_1BForCausalLM(model=model, tokenizer=tokenizer, runtime_config=runtime)
    assert model.model_args is runtime
    assert product.generation_config.stop_token_ids == (128001,)
    assert product.model_name == "model"
    assert product.model_cache_path is None
    assert product.max_seq_len == 4096
    assert product.max_context_len == 131072


def test_hf_attention_and_mlp_weights_match_reference_layouts():
    hidden_size = 128
    num_attention_heads = 32
    num_key_value_heads = 8
    num_devices = 8
    head_dim = hidden_size // num_attention_heads
    kv_width = num_key_value_heads * head_dim
    config = SimpleNamespace(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_size=hidden_size,
    )
    q = torch.arange(hidden_size * hidden_size, dtype=torch.float32).reshape(hidden_size, hidden_size)
    k = torch.arange(kv_width * hidden_size, dtype=torch.float32).reshape(kv_width, hidden_size) + 100_000
    v = k + 100_000
    o = q + 300_000
    attention = SimpleNamespace(
        config=config,
        q_proj=SimpleNamespace(weight=q),
        k_proj=SimpleNamespace(weight=k),
        v_proj=SimpleNamespace(weight=v),
        o_proj=SimpleNamespace(weight=o),
    )

    wqkv, wo = weight_utils.attention_wqkv_wo_from_hf_layer(attention, num_devices=num_devices)
    q_meta = q.view(num_attention_heads, 2, head_dim // 2, hidden_size).transpose(1, 2).reshape(q.shape).T
    k_meta = k.view(num_key_value_heads, 2, head_dim // 2, hidden_size).transpose(1, 2).reshape(k.shape).T
    expected_qkv = (
        torch.cat(
            [
                torch.cat(parts, dim=-1)
                for parts in zip(
                    torch.chunk(q_meta, num_devices, dim=1),
                    torch.chunk(k_meta, num_devices, dim=1),
                    torch.chunk(v.T, num_devices, dim=1),
                )
            ],
            dim=-1,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    assert wqkv.shape == (1, 1, hidden_size, hidden_size + 2 * kv_width)
    torch.testing.assert_close(wqkv, expected_qkv)
    torch.testing.assert_close(wo, o.T.unsqueeze(0).unsqueeze(0))

    gate = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    down = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    up = gate + 100
    mlp = SimpleNamespace(
        gate_proj=SimpleNamespace(weight=gate),
        down_proj=SimpleNamespace(weight=down),
        up_proj=SimpleNamespace(weight=up),
    )
    w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(mlp)
    torch.testing.assert_close(w1, gate.T)
    torch.testing.assert_close(w2, down.T)
    torch.testing.assert_close(w3, up.T)


def test_hf_rope_tables_match_real_llama32_scaled_rotary_reference():
    head_dim = 64
    table_len = LLAMA32_ROPE_PARAMETERS["original_max_position_embeddings"] + 128
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=head_dim,
        max_position_embeddings=131072,
        rope_parameters=LLAMA32_ROPE_PARAMETERS,
    )
    rotary = LlamaRotaryEmbedding(config)

    cos, sin = weight_utils.build_rope_cos_sin_torch(
        rotary, table_len=table_len, head_dim=head_dim, dtype=torch.bfloat16
    )
    x = torch.zeros(1, 1, table_len, head_dim, dtype=torch.bfloat16)
    position_ids = torch.arange(table_len, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        hf_cos, hf_sin = rotary(x, position_ids)
    expected_cos = hf_cos.float().squeeze(0)[:, : head_dim // 2].repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0)
    expected_sin = hf_sin.float().squeeze(0)[:, : head_dim // 2].repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0)

    assert config.rope_parameters == LLAMA32_ROPE_PARAMETERS
    assert cos.shape == sin.shape == (1, 1, table_len, head_dim)
    assert cos.dtype == sin.dtype == torch.bfloat16
    torch.testing.assert_close(cos, expected_cos.to(torch.bfloat16))
    torch.testing.assert_close(sin, expected_sin.to(torch.bfloat16))


def test_convert_hf_model_weights_covers_real_nonempty_llama_layer():
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=4,
        vocab_size=128,
        max_position_embeddings=131072,
        rope_parameters=LLAMA32_ROPE_PARAMETERS,
        tie_word_embeddings=True,
    )
    hf = LlamaForCausalLM(config).eval()
    weights = convert_hf_model_weights(
        hf,
        config,
        n_layers=1,
        num_devices=8,
        rope_table_len=128,
        head_dim=4,
    )

    assert len(weights.layers) == 1
    layer_weights = weights.layers[0]
    assert layer_weights.wqkv.shape == (1, 1, 128, 192)
    assert layer_weights.wo.shape == (1, 1, 128, 128)
    assert layer_weights.w1.shape == (128, 256)
    assert layer_weights.w2.shape == (256, 128)
    assert layer_weights.w3.shape == (128, 256)
    assert layer_weights.attention_norm.shape == layer_weights.ff_norm.shape == (128,)
    assert weights.embedding.shape == (1, 1, 128, 128)
    assert weights.rope_cos.shape == weights.rope_sin.shape == (1, 1, 128, 4)
    assert weights.final_norm.shape == (128,)
    torch.testing.assert_close(weights.lm_head, hf.model.embed_tokens.weight.detach().to(torch.bfloat16))


def test_tied_embedding_is_explicit_lm_head_construction_source():
    class Rotary:
        def __call__(self, x, position_ids):
            return torch.ones(1, position_ids.shape[-1], x.shape[-1]), torch.zeros(
                1, position_ids.shape[-1], x.shape[-1]
            )

    tied_weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    decoy_lm_head = torch.full((6, 4), -99.0)
    base = SimpleNamespace(
        embed_tokens=SimpleNamespace(weight=tied_weight),
        rotary_emb=Rotary(),
        layers=[],
        norm=SimpleNamespace(weight=torch.ones(4)),
    )
    hf = SimpleNamespace(model=base, lm_head=SimpleNamespace(weight=decoy_lm_head))
    config = SimpleNamespace(tie_word_embeddings=True)
    weights = convert_hf_model_weights(
        hf,
        config,
        n_layers=0,
        num_devices=1,
        rope_table_len=8,
        head_dim=4,
    )

    torch.testing.assert_close(weights.lm_head, tied_weight.to(torch.bfloat16))
    assert not torch.equal(weights.lm_head, decoy_lm_head.to(torch.bfloat16))
    assert weights.embedding.shape == (1, 1, 6, 4)


def test_config_builder_is_owned_by_model_module():
    assert hf_adaptor.build_llama32_1b_transformer_1d_config is llama_model.build_llama32_1b_transformer_1d_config
    assert llama_model.build_llama32_1b_transformer_1d_config.__module__ == llama_model.__name__


def test_post_attention_norm_program_and_memory_use_same_mlp_grid(monkeypatch):
    grid = SimpleNamespace(num_cores=64)
    program = object()
    memory = object()
    captured = {}

    monkeypatch.setattr(llama_model, "get_padded_hidden_dim", lambda *_: 8192)
    monkeypatch.setattr(llama_model, "_dram_shard_core_grid_k_n", lambda *_: grid)
    monkeypatch.setattr(
        llama_model,
        "_create_sharded_norm_program_config",
        lambda dim, selected_grid, rows, tile: captured.update(program=(dim, selected_grid, rows, tile)) or program,
    )
    monkeypatch.setattr(
        llama_model.ttnn,
        "create_sharded_memory_config",
        lambda shape, selected_grid, *args, **kwargs: captured.update(memory=(shape, selected_grid)) or memory,
    )

    assert llama_model._post_attn_norm_decode_configs(
        dim=2048,
        hidden_dim=8192,
        num_devices=1,
        max_batch_size=1,
    ) == (program, memory)
    assert captured["program"] == (2048, grid, 32, 32)
    assert captured["memory"] == ((32, 32), grid)
