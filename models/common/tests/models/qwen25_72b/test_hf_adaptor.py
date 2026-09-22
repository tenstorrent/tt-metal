# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from models.common.models.qwen25_72b import generator, hf_adaptor, weight_utils
from models.common.models.qwen25_72b.hf_adaptor import (
    Qwen25_72BForCausalLM,
    Qwen25_72BRuntimeConfig,
    _trace_seq_lens,
    convert_hf_model_weights,
)


def _runtime_config():
    return Qwen25_72BRuntimeConfig(
        model_name="Qwen2.5-72B-Instruct",
        model_cache_path=None,
        max_prefill_chunk_size=2048,
        max_context_len=131072,
        max_seq_len=4096,
        trace_prefill_supported_seq_lens=(128, 1024),
        n_layers=80,
        n_kv_heads=8,
        head_dim=128,
        max_batch_size=32,
        cluster_shape=[1, 8],
    )


def test_runtime_config_preserves_t3k_trace_and_batched_prefill_policy():
    runtime = _runtime_config()
    assert runtime.can_enable_trace(128)
    assert runtime.can_enable_trace(1024, num_cached_tokens=32)
    assert not runtime.can_enable_trace(2048)
    assert runtime.supports_batched_prefill
    assert runtime.max_prefill_batch_size == 32
    assert runtime.batched_prefill_batched_extract


def test_pinned_revision_is_provider_and_generator_default():
    expected = "495f39366efef23836d0cfae4fbe635880d2be31"
    assert hf_adaptor.DEFAULT_HF_REVISION == expected
    assert generator.Qwen25_72BGeneratorConfig.__dataclass_fields__["hf_revision"].default == expected


def test_trace_policy_is_tp8_only_and_keeps_128_and_1024(expect_error):
    assert _trace_seq_lens(8, 2048, 4096) == (128, 1024)
    for devices in (1, 2, 4):
        with expect_error(ValueError, "exactly 8 devices"):
            _trace_seq_lens(devices, 2048, 4096)


def test_product_binds_runtime_config_and_qwen_stop_tokens():
    model = SimpleNamespace(config=SimpleNamespace(max_seq_len=4096), model_args=None)
    tokenizer = SimpleNamespace(stop_tokens=[151643, 151644])
    product = Qwen25_72BForCausalLM(model=model, tokenizer=tokenizer, runtime_config=_runtime_config())
    assert model.model_args is product.runtime_config
    assert product.generation_config.stop_token_ids == (151643, 151644)
    assert product.max_seq_len == 4096
    assert product.max_context_len == 131072


def test_qwen_stop_tokens_include_turn_terminators():
    token_map = {"<|im_end|>": 151645, "<|im_start|>": 151644}
    tokenizer = SimpleNamespace(
        eos_token_id=151643,
        convert_tokens_to_ids=lambda token: token_map.get(token, -1),
    )
    assert hf_adaptor._qwen_stop_token_ids(tokenizer) == (151643, 151645, 151644)


def test_qkv_weights_and_bias_use_reverse_permutation_and_device_major_packing():
    hidden_size = 16
    n_heads = 4
    n_kv_heads = 2
    head_dim = 4
    num_devices = 2
    kv_width = n_kv_heads * head_dim
    q = torch.arange(hidden_size * hidden_size, dtype=torch.float32).reshape(hidden_size, hidden_size)
    k = torch.arange(kv_width * hidden_size, dtype=torch.float32).reshape(kv_width, hidden_size) + 10_000
    v = k + 10_000
    o = q + 30_000
    bq = torch.arange(hidden_size, dtype=torch.float32)
    bk = torch.arange(kv_width, dtype=torch.float32) + 100
    bv = torch.arange(kv_width, dtype=torch.float32) + 200
    attention = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
        ),
        q_proj=SimpleNamespace(weight=q, bias=bq),
        k_proj=SimpleNamespace(weight=k, bias=bk),
        v_proj=SimpleNamespace(weight=v, bias=bv),
        o_proj=SimpleNamespace(weight=o),
    )

    wqkv, wo, q_norm, k_norm, bias = weight_utils.attention_wqkv_wo_from_hf_layer(attention, num_devices)
    q_meta = weight_utils.reverse_permute(q, n_heads, hidden_size, hidden_size).T
    k_meta = weight_utils.reverse_permute(k, n_kv_heads, kv_width, hidden_size).T
    bq_meta = weight_utils.reverse_permute_1d(bq.view(n_heads, head_dim)).view(-1)
    bk_meta = weight_utils.reverse_permute_1d(bk.view(n_kv_heads, head_dim)).view(-1)
    expected_weights = (
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
    expected_bias = torch.cat(
        [
            torch.cat(parts, dim=-1)
            for parts in zip(
                torch.chunk(bq_meta, num_devices),
                torch.chunk(bk_meta, num_devices),
                torch.chunk(bv, num_devices),
            )
        ]
    )

    torch.testing.assert_close(wqkv, expected_weights)
    torch.testing.assert_close(wo, o.T.unsqueeze(0).unsqueeze(0))
    torch.testing.assert_close(bias, expected_bias)
    assert q_norm is None and k_norm is None


def test_hf_rope_tables_preserve_plain_theta_one_million():
    head_dim = 16
    table_len = 128
    config = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
    )
    rotary = Qwen2RotaryEmbedding(config)
    cos, sin = weight_utils.build_rope_cos_sin_torch(rotary, table_len, head_dim, torch.bfloat16)
    x = torch.zeros(1, 1, table_len, head_dim, dtype=torch.bfloat16)
    positions = torch.arange(table_len).unsqueeze(0)
    with torch.no_grad():
        hf_cos, hf_sin = rotary(x, positions)
    expected_cos, expected_sin = weight_utils.permute_hf_rope_to_meta_tables(hf_cos.float(), hf_sin.float())
    assert config.rope_parameters["rope_theta"] == 1_000_000.0
    torch.testing.assert_close(cos, expected_cos.to(torch.bfloat16))
    torch.testing.assert_close(sin, expected_sin.to(torch.bfloat16))


def test_convert_hf_model_weights_covers_real_nonempty_qwen_layer():
    config = Qwen2Config(
        hidden_size=256,
        intermediate_size=320,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=8,
        vocab_size=128,
        max_position_embeddings=32768,
        tie_word_embeddings=False,
    )
    hf = Qwen2ForCausalLM(config).eval()
    weights = convert_hf_model_weights(
        hf,
        config,
        n_layers=1,
        num_devices=8,
        rope_table_len=128,
        head_dim=4,
    )
    assert len(weights.layers) == 1
    assert weights.layers[0].wqkv.shape == (1, 1, 256, 320)
    assert weights.layers[0].wqkv_bias.shape == (320,)
    assert weights.lm_head.shape == (128, 256)
