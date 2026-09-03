# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from models.common.models.qwen25_7b import hf_adaptor
from models.common.models.qwen25_7b import model as qwen_model
from models.common.models.qwen25_7b import weight_utils
from models.common.models.qwen25_7b.hf_adaptor import Qwen25ForCausalLM as Qwen25Product
from models.common.models.qwen25_7b.hf_adaptor import Qwen25RuntimeConfig, _trace_seq_lens, convert_hf_model_weights


def test_runtime_config_preserves_tp2_trace_and_batched_prefill_policy():
    runtime = Qwen25RuntimeConfig(
        model_name="Qwen2.5-7B-Instruct",
        model_cache_path=None,
        max_prefill_chunk_size=2048,
        max_context_len=32768,
        max_seq_len=4096,
        trace_prefill_supported_seq_lens=(128, 1024),
    )
    assert runtime.can_enable_trace(128, num_cached_tokens=32)
    assert runtime.can_enable_trace(1024)
    assert not runtime.can_enable_trace(2048)
    assert runtime.supports_batched_prefill
    assert runtime.max_prefill_batch_size == 8
    assert runtime.batched_prefill_batched_extract
    assert _trace_seq_lens(2, 2048, 4096) == (128, 1024)


def test_provider_rejects_non_tp2_before_loading_hf(expect_error):
    mesh = SimpleNamespace(get_num_devices=lambda: 1)
    with expect_error(ValueError, "logical TP2 lanes only"):
        hf_adaptor.from_pretrained(mesh)


def test_product_binds_runtime_config_and_stop_tokens():
    model = SimpleNamespace(config=SimpleNamespace(max_seq_len=4096), model_args=None)
    tokenizer = SimpleNamespace(stop_tokens=[151643, 151644])
    runtime = Qwen25RuntimeConfig(
        model_name="model",
        model_cache_path=None,
        max_prefill_chunk_size=2048,
        max_context_len=32768,
        max_seq_len=4096,
        trace_prefill_supported_seq_lens=(128, 1024),
    )
    product = Qwen25Product(model=model, tokenizer=tokenizer, runtime_config=runtime)
    assert model.model_args is runtime
    assert product.generation_config.stop_token_ids == (151643, 151644)
    assert product.max_seq_len == 4096
    assert product.max_context_len == 32768


def test_tokenizer_adds_eos_and_im_start_and_threads_revision(monkeypatch):
    tokenizer = SimpleNamespace(
        eos_token_id=151643,
        convert_tokens_to_ids=lambda token: 151644 if token == "<|im_start|>" else -1,
    )
    seen = {}

    def fake_from_pretrained(model, **kwargs):
        seen.update(model=model, **kwargs)
        return tokenizer

    monkeypatch.setattr(hf_adaptor.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    assert hf_adaptor.load_tokenizer("Qwen/Qwen2.5-7B-Instruct", "revision") is tokenizer
    assert tokenizer.stop_tokens == [151643, 151644]
    assert seen["revision"] == "revision"


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


def test_conversion_covers_qkv_bias_and_untied_lm_head():
    config = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=32768,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
        tie_word_embeddings=False,
    )
    hf = Qwen2ForCausalLM(config).eval()
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
    assert layer.wqkv_bias.shape == (128,)
    assert layer.wo.shape == (1, 1, 64, 64)
    assert layer.w1.shape == layer.w3.shape == (64, 128)
    assert layer.w2.shape == (128, 64)
    torch.testing.assert_close(weights.lm_head, hf.lm_head.weight.detach().to(torch.bfloat16))
    assert weights.lm_head.data_ptr() != weights.embedding.data_ptr()


def test_config_builder_is_owned_by_model_module():
    assert hf_adaptor.build_qwen25_7b_transformer_config is qwen_model.build_qwen25_7b_transformer_config
    assert qwen_model.build_qwen25_7b_transformer_config.__module__ == qwen_model.__name__


def test_post_attention_norm_program_and_memory_use_same_mlp_grid(monkeypatch):
    grid = SimpleNamespace(num_cores=28)
    program = object()
    memory = object()
    captured = {}

    monkeypatch.setattr(qwen_model, "get_padded_hidden_dim", lambda *_: 18944)
    monkeypatch.setattr(qwen_model, "_dram_shard_core_grid_k_n", lambda *_: grid)
    monkeypatch.setattr(
        qwen_model,
        "_create_sharded_norm_program_config",
        lambda dim, selected_grid, rows, tile: captured.update(program=(dim, selected_grid, rows, tile)) or program,
    )
    monkeypatch.setattr(
        qwen_model.ttnn,
        "create_sharded_memory_config",
        lambda shape, selected_grid, *args, **kwargs: captured.update(memory=(shape, selected_grid)) or memory,
    )

    assert qwen_model._post_attn_norm_decode_configs(
        dim=3584,
        hidden_dim=18944,
        num_devices=2,
        max_batch_size=32,
    ) == (program, memory)
    assert captured["program"] == (3584, grid, 32, 32)
    assert captured["memory"] == ((32, 128), grid)


def test_decoder_layer_prefill_calls_chunk_capable_attention_entrypoint(monkeypatch):
    captured = {}
    attention_output = object()
    final_output = object()
    attention = SimpleNamespace(
        prefill_forward=lambda x, rot_mats, **kwargs: captured.update(attention=(x, rot_mats, kwargs))
        or attention_output
    )
    layer = qwen_model.Qwen25_7BDecoderLayer(
        input_layernorm=SimpleNamespace(prefill_forward=lambda x: x),
        self_attn=attention,
        post_attention_layernorm=SimpleNamespace(prefill_forward=lambda x: x),
        mlp=SimpleNamespace(prefill_forward=lambda x: x),
    )
    monkeypatch.setattr(qwen_model, "_all_gather_rmsnorm_tensor", lambda _norm, x: x)
    monkeypatch.setattr(
        qwen_model.ttnn,
        "add",
        lambda *_args, **_kwargs: final_output,
    )

    chunk_start_idx_tensor = object()
    rot_mats = (object(), object())
    assert (
        layer.prefill_forward(
            object(),
            rot_mats,
            user_id=[0, 1],
            page_table=object(),
            chunk_page_table=object(),
            chunk_start_idx=128,
            batch_size=2,
            chunk_start_idx_tensor=chunk_start_idx_tensor,
        )
        is final_output
    )
    assert captured["attention"][1] is rot_mats
    assert captured["attention"][2]["chunk_start_idx_tensor"] is chunk_start_idx_tensor
    assert captured["attention"][2]["batch_size"] == 2
