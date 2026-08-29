# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from models.common.models.qwen3_32b import generator, hf_adaptor, weight_utils
from models.common.models.qwen3_32b.hf_adaptor import Qwen3_32BForCausalLM, Qwen3_32BRuntimeConfig, _trace_seq_lens


def _runtime_config():
    return Qwen3_32BRuntimeConfig(
        model_name="Qwen/Qwen3-32B",
        model_cache_path=None,
        max_prefill_chunk_size=4096,
        max_context_len=40960,
        max_seq_len=4096,
        trace_prefill_supported_seq_lens=(128, 1024),
        n_layers=64,
        n_kv_heads=8,
        head_dim=128,
        max_batch_size=32,
        cluster_shape=[1, 8],
    )


def test_runtime_config_preserves_t3k_trace_and_batched_prefill_policy():
    runtime = _runtime_config()
    assert runtime.can_enable_trace(128)
    assert runtime.can_enable_trace(1024, num_cached_tokens=0)
    assert not runtime.can_enable_trace(1024, num_cached_tokens=32)
    assert not runtime.can_enable_trace(2048)
    assert runtime.supports_batched_prefill
    assert runtime.max_prefill_batch_size == 32
    assert runtime.batched_prefill_batched_extract


def test_pinned_revision_is_provider_and_generator_default():
    expected = "9216db5781bf21249d130ec9da846c4624c16137"
    assert hf_adaptor.DEFAULT_HF_REVISION == expected
    assert generator.Qwen3_32BGeneratorConfig.__dataclass_fields__["hf_revision"].default == expected


def test_trace_policy_is_tp8_only_and_keeps_128_and_1024(expect_error):
    assert _trace_seq_lens(8, 4096, 4096) == (128, 1024)
    for devices in (1, 2, 4):
        with expect_error(ValueError, "exactly 8 devices"):
            _trace_seq_lens(devices, 4096, 4096)


def test_product_binds_runtime_config_and_qwen_stop_tokens():
    model = SimpleNamespace(config=SimpleNamespace(max_seq_len=4096), model_args=None)
    tokenizer = SimpleNamespace(stop_tokens=[151645, 151644])
    product = Qwen3_32BForCausalLM(model=model, tokenizer=tokenizer, runtime_config=_runtime_config())
    assert model.model_args is product.runtime_config
    assert product.generation_config.stop_token_ids == (151645, 151644)
    assert product.max_seq_len == 4096
    assert product.max_context_len == 40960


def test_qwen_stop_tokens_include_turn_terminators():
    token_map = {"<|im_end|>": 151645, "<|im_start|>": 151644}
    tokenizer = SimpleNamespace(
        eos_token_id=151643,
        convert_tokens_to_ids=lambda token: token_map.get(token, -1),
    )
    assert hf_adaptor._qwen_stop_token_ids(tokenizer) == (151643, 151645, 151644)


def test_qwen3_qkv_weights_preserve_qk_norm_no_bias_and_explicit_head_dim():
    hidden_size = 8
    n_heads = 4
    n_kv_heads = 2
    head_dim = 4
    num_devices = 2
    q_width = n_heads * head_dim
    kv_width = n_kv_heads * head_dim
    q = torch.arange(q_width * hidden_size, dtype=torch.float32).reshape(q_width, hidden_size)
    k = torch.arange(kv_width * hidden_size, dtype=torch.float32).reshape(kv_width, hidden_size) + 10_000
    v = k + 10_000
    o = torch.arange(hidden_size * q_width, dtype=torch.float32).reshape(hidden_size, q_width) + 30_000
    q_norm = torch.arange(head_dim, dtype=torch.float32) + 1
    k_norm = q_norm + 10
    attention = SimpleNamespace(
        head_dim=head_dim,
        config=SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=head_dim,
        ),
        q_proj=SimpleNamespace(weight=q, bias=None),
        k_proj=SimpleNamespace(weight=k, bias=None),
        v_proj=SimpleNamespace(weight=v, bias=None),
        o_proj=SimpleNamespace(weight=o),
        q_norm=SimpleNamespace(weight=q_norm),
        k_norm=SimpleNamespace(weight=k_norm),
    )

    wqkv, wo, qn, kn, bias = weight_utils.attention_wqkv_wo_from_hf_layer(attention, num_devices)

    assert wqkv.shape == (1, 1, hidden_size, (q_width + kv_width + kv_width))
    assert wo.shape == (1, 1, q_width, hidden_size)
    assert torch.equal(qn, weight_utils.reverse_permute_1d(q_norm))
    assert torch.equal(kn, weight_utils.reverse_permute_1d(k_norm))
    assert bias is None


def test_qwen3_lm_head_vocab_padding_masks_real_vocab_tail():
    assert weight_utils.lm_head_padded_vocab_size(151936, 8) == 152064
