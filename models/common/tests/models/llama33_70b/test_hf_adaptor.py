# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from models.common.models.llama33_70b import hf_adaptor
from models.common.models.llama33_70b import model as llama_model
from models.common.models.llama33_70b import weight_utils
from models.common.models.llama33_70b.hf_adaptor import (
    Llama33_70BForCausalLM,
    Llama33_70BRuntimeConfig,
    convert_hf_model_weights,
)

LLAMA33_ROPE_PARAMETERS = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_theta": 500000.0,
}


def _runtime_config():
    return Llama33_70BRuntimeConfig(
        model_name="Llama-3.3-70B-Instruct",
        model_cache_path=None,
        max_prefill_chunk_size=2048,
        max_context_len=131072,
        max_seq_len=4096,
        trace_prefill_supported_seq_lens=(128, 2048),
        trace_prefill_warmup_seq_lens=(128, 2048, 4096),
    )


def test_runtime_config_preserves_t3k_trace_and_batched_prefill_policy():
    runtime = _runtime_config()
    assert runtime.can_enable_trace(128)
    assert runtime.can_enable_trace(128, num_cached_tokens=32)
    assert runtime.can_enable_trace(2048)
    assert not runtime.can_enable_trace(1024)
    assert not runtime.can_enable_trace(4096)
    assert runtime.supports_batched_prefill
    assert runtime.max_prefill_batch_size == 32
    assert runtime.batched_prefill_batched_extract


def test_trace_policy_is_tp8_only_and_includes_fixed_chunk_invocation(expect_error):
    assert hf_adaptor._trace_seq_lens(8, 2048, 4096) == (128, 2048)
    assert hf_adaptor._trace_warmup_seq_lens(2048, 4096) == (128, 2048, 4096)
    for devices in (1, 2, 4):
        with expect_error(ValueError, "exactly 8 devices"):
            hf_adaptor._trace_seq_lens(devices, 2048, 4096)


def test_product_binds_runtime_and_preserves_all_llama3_stop_ids():
    model = SimpleNamespace(config=SimpleNamespace(max_seq_len=4096), model_args=None)
    tokenizer = SimpleNamespace(stop_tokens=[128001, 128008, 128009])
    product = Llama33_70BForCausalLM(model=model, tokenizer=tokenizer, runtime_config=_runtime_config())
    assert model.model_args is product.runtime_config
    assert product.generation_config.stop_token_ids == (128001, 128008, 128009)
    assert product.max_seq_len == 4096
    assert product.max_context_len == 131072


def test_post_attention_norm_decode_uses_mlp_input_grid():
    program_config, memory_config = llama_model._post_attn_norm_decode_configs(
        dim=8192,
        hidden_dim=28672,
        num_devices=8,
        max_batch_size=32,
    )

    assert str(program_config.compute_with_storage_grid_size) == "8-2"
    assert '"end":{"x":7,"y":1}' in str(memory_config)
    assert "shape=[32, 512]" in str(memory_config)


def test_all_gather_rmsnorm_honors_memory_config_when_tensor_is_already_full_width(monkeypatch):
    requested_memory_config = object()
    converted_tensor = object()
    x = SimpleNamespace(shape=(1, 1, 32, 8192))
    norm = SimpleNamespace(
        config=SimpleNamespace(
            mesh_device=SimpleNamespace(get_num_devices=lambda: 8),
            weight=SimpleNamespace(source=SimpleNamespace(numel=lambda: 8192)),
        )
    )
    calls = []

    def fake_to_memory_config(tensor, memory_config):
        calls.append((tensor, memory_config))
        return converted_tensor

    monkeypatch.setattr(llama_model.ttnn, "to_memory_config", fake_to_memory_config)

    assert llama_model._all_gather_rmsnorm_tensor(norm, x, memory_config=requested_memory_config) is converted_tensor
    assert calls == [(x, requested_memory_config)]


def test_hf_attention_and_mlp_weights_match_llama33_reference_layouts():
    # Reduced tensors preserve Llama-3.3's 64Q/8KV head topology and TP8 packing.
    hidden_size = 256
    num_attention_heads = 64
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


def test_hf_rope_tables_match_real_llama33_factor8_scaled_rotary_reference():
    head_dim = 16
    table_len = LLAMA33_ROPE_PARAMETERS["original_max_position_embeddings"] + 128
    config = LlamaConfig(
        hidden_size=384,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=24,
        num_key_value_heads=8,
        head_dim=head_dim,
        max_position_embeddings=131072,
        rope_parameters=LLAMA33_ROPE_PARAMETERS,
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

    assert config.rope_parameters == LLAMA33_ROPE_PARAMETERS
    assert cos.shape == sin.shape == (1, 1, table_len, head_dim)
    assert cos.dtype == sin.dtype == torch.bfloat16
    torch.testing.assert_close(cos, expected_cos.to(torch.bfloat16))
    torch.testing.assert_close(sin, expected_sin.to(torch.bfloat16))


def test_convert_hf_model_weights_covers_real_nonempty_llama33_layer():
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=320,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=4,
        vocab_size=128,
        max_position_embeddings=131072,
        rope_parameters=LLAMA33_ROPE_PARAMETERS,
        tie_word_embeddings=False,
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
    assert layer_weights.wqkv.shape == (1, 1, 256, 320)
    assert layer_weights.wo.shape == (1, 1, 256, 256)
    assert layer_weights.w1.shape == (256, 320)
    assert layer_weights.w2.shape == (320, 256)
    assert layer_weights.w3.shape == (256, 320)
    assert layer_weights.attention_norm.shape == layer_weights.ff_norm.shape == (256,)
    assert weights.embedding.shape == (1, 1, 128, 256)
    assert weights.rope_cos.shape == weights.rope_sin.shape == (1, 1, 128, 4)
    assert weights.final_norm.shape == (256,)
    torch.testing.assert_close(weights.lm_head, hf.lm_head.weight.detach().to(torch.bfloat16))


def test_untied_lm_head_is_explicit_conversion_source():
    class Rotary:
        def __call__(self, x, position_ids):
            return torch.ones(1, position_ids.shape[-1], x.shape[-1]), torch.zeros(
                1, position_ids.shape[-1], x.shape[-1]
            )

    embedding_weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    lm_head_weight = embedding_weight + 100
    base = SimpleNamespace(
        embed_tokens=SimpleNamespace(weight=embedding_weight),
        rotary_emb=Rotary(),
        layers=[],
        norm=SimpleNamespace(weight=torch.ones(4)),
    )
    hf = SimpleNamespace(model=base, lm_head=SimpleNamespace(weight=lm_head_weight))
    weights = convert_hf_model_weights(
        hf,
        SimpleNamespace(tie_word_embeddings=False),
        n_layers=0,
        num_devices=8,
        rope_table_len=8,
        head_dim=4,
    )

    torch.testing.assert_close(weights.lm_head, lm_head_weight.to(torch.bfloat16))
    assert not torch.equal(weights.lm_head, embedding_weight.to(torch.bfloat16))


def test_tokenizer_preserves_scalar_and_generation_eos_ids(monkeypatch):
    tokenizer = SimpleNamespace(eos_token_id=[128001, 128008, 128009])
    monkeypatch.setattr(hf_adaptor.AutoTokenizer, "from_pretrained", lambda *_, **__: tokenizer)
    assert hf_adaptor.load_tokenizer("meta-llama/Llama-3.3-70B-Instruct") is tokenizer
    assert tokenizer.stop_tokens == [128001, 128008, 128009]


def test_hf_generation_stop_ids_are_deduplicated_in_order():
    hf = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[128001, 128008, 128009, 128001]))
    assert hf_adaptor._stop_token_ids(hf) == (128001, 128008, 128009)


def test_encode_prompt_uses_the_provider_chat_template():
    calls = []
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: calls.append((messages, kwargs)) or [101, 102, 103]
    )
    assert hf_adaptor.encode_prompt(tokenizer, "Hello") == [101, 102, 103]
    assert calls == [
        (
            [{"role": "user", "content": "Hello"}],
            {"add_generation_prompt": True, "tokenize": True},
        )
    ]


def test_config_builder_is_owned_by_model_module():
    assert hf_adaptor.build_llama33_70b_transformer_1d_config is llama_model.build_llama33_70b_transformer_1d_config
    assert llama_model.build_llama33_70b_transformer_1d_config.__module__ == llama_model.__name__
