# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import inspect
import json
from pathlib import Path

import pytest
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding, apply_rotary_pos_emb

import ttnn
from models.autoports.tiiuae_falcon3_7b_base.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    IR_REPRESENTATIVE_LAYER,
    FunctionalDecoder,
)
from models.common.utility_functions import comp_pcc

HF_MODEL = "tiiuae/Falcon3-7B-Base"
REAL_WEIGHT_PCC = 0.99
SYNTHETIC_PCC = 0.99


def _config():
    config = AutoConfig.from_pretrained(HF_MODEL, local_files_only=True)
    config._attn_implementation = "eager"
    return config


def _synthetic_state_dict(config, layer_idx: int):
    generator = torch.Generator().manual_seed(20260715)

    def weight(shape):
        tensor = torch.empty(shape, dtype=torch.bfloat16)
        return tensor.normal_(mean=0.0, std=0.02, generator=generator)

    prefix = f"model.layers.{layer_idx}."
    hidden = config.hidden_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim
    intermediate = config.intermediate_size
    return {
        prefix + "self_attn.q_proj.weight": weight((q_width, hidden)),
        prefix + "self_attn.k_proj.weight": weight((kv_width, hidden)),
        prefix + "self_attn.v_proj.weight": weight((kv_width, hidden)),
        prefix + "self_attn.o_proj.weight": weight((hidden, hidden)),
        prefix + "mlp.gate_proj.weight": weight((intermediate, hidden)),
        prefix + "mlp.up_proj.weight": weight((intermediate, hidden)),
        prefix + "mlp.down_proj.weight": weight((hidden, intermediate)),
        prefix + "input_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        prefix + "post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
    }


def _real_layer_state_dict(layer_idx: int):
    index_path = Path(hf_hub_download(HF_MODEL, "model.safetensors.index.json"))
    weight_map = json.loads(index_path.read_text())["weight_map"]
    prefix = f"model.layers.{layer_idx}."
    keys = sorted(key for key in weight_map if key.startswith(prefix))
    shards = sorted({weight_map[key] for key in keys})
    state_dict = {}
    for shard in shards:
        shard_path = hf_hub_download(HF_MODEL, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as tensors:
            state_dict.update({key: tensors.get_tensor(key) for key in keys if key in tensors.keys()})
    assert set(state_dict) == set(keys)
    return state_dict


def _hf_layer(config, state_dict, layer_idx: int):
    prefix = f"model.layers.{layer_idx}."
    local_state = {key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)}
    with torch.device("meta"):
        layer = LlamaDecoderLayer(config, layer_idx=layer_idx)
    layer.load_state_dict(local_state, strict=True, assign=True)
    return layer.eval()


def _position_embeddings(config, hidden_states, start: int = 0):
    positions = torch.arange(start, start + hidden_states.shape[1], dtype=torch.long)
    positions = positions.unsqueeze(0).expand(hidden_states.shape[0], -1)
    rotary = LlamaRotaryEmbedding(config)
    return rotary(hidden_states, positions)


def _causal_mask(seq_len: int, dtype):
    mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(mask, diagonal=1)


@torch.no_grad()
def _hf_prefill(layer, config, hidden_states, cache=None):
    return layer(
        hidden_states,
        attention_mask=_causal_mask(hidden_states.shape[1], hidden_states.dtype),
        past_key_values=cache,
        use_cache=cache is not None,
        position_embeddings=_position_embeddings(config, hidden_states),
    )


@torch.no_grad()
def _hf_decode(layer, config, hidden_states, cache, position: int):
    attention_mask = torch.zeros((1, 1, 1, position + 1), dtype=hidden_states.dtype)
    return layer(
        hidden_states,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        position_embeddings=_position_embeddings(config, hidden_states, start=position),
    )


@torch.no_grad()
def _hf_decode_query(layer, config, hidden_states, position: int):
    normed = layer.input_layernorm(hidden_states)
    query = layer.self_attn.q_proj(normed)
    query = query.view(
        hidden_states.shape[0],
        hidden_states.shape[1],
        config.num_attention_heads,
        config.head_dim,
    ).transpose(1, 2)
    key = layer.self_attn.k_proj(normed)
    key = key.view(
        hidden_states.shape[0],
        hidden_states.shape[1],
        config.num_key_value_heads,
        config.head_dim,
    ).transpose(1, 2)
    cos, sin = _position_embeddings(config, hidden_states, start=position)
    query, _ = apply_rotary_pos_emb(query, key, cos, sin)
    return query.squeeze(2)


def _tt_input(hidden_states, device):
    return ttnn.from_torch(
        hidden_states.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _assert_pcc(name: str, expected, actual, threshold: float):
    passed, pcc = comp_pcc(expected.float(), actual.float(), pcc=threshold)
    logger.info(f"{name}: PCC={pcc:.8f} target={threshold}")
    assert passed, f"{name}: PCC={pcc:.8f} below {threshold}"
    return pcc


def _assert_cache_prefix_matches_hf(hf_cache, key_cache, value_cache, seq_len: int):
    hf_layer_cache = hf_cache.layers[IR_REPRESENTATIVE_LAYER]
    assert hf_layer_cache.is_initialized
    tt_key = ttnn.to_torch(key_cache)[:, :, :seq_len, :]
    tt_value = ttnn.to_torch(value_cache)[:, :, :seq_len, :]
    _assert_pcc("synthetic prefill key cache", hf_layer_cache.keys, tt_key, SYNTHETIC_PCC)
    _assert_pcc("synthetic prefill value cache", hf_layer_cache.values, tt_value, SYNTHETIC_PCC)


def _release_model(model):
    for name in (
        "qkv_weight",
        "o_weight",
        "gate_weight",
        "up_weight",
        "down_weight",
        "input_norm_weight",
        "post_attention_norm_weight",
        "cos_cache",
        "sin_cache",
        "decode_positions",
    ):
        getattr(model, name).deallocate(True)
    del model
    gc.collect()


def _run_prefill(model, device, hidden_states):
    key_cache, value_cache = model.allocate_kv_cache()
    tt_hidden = _tt_input(hidden_states, device)
    tt_output = model.prefill_forward(tt_hidden, key_cache=key_cache, value_cache=value_cache)
    output = ttnn.to_torch(tt_output).squeeze(0)
    tt_hidden.deallocate(True)
    tt_output.deallocate(True)
    return output, key_cache, value_cache


def _run_decode(model, device, hidden_states, key_cache, value_cache, position: int):
    tt_hidden = _tt_input(hidden_states, device)
    cache_position = ttnn.from_torch(
        torch.full((EMITTED_BATCH,), position, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = model.decode_forward(
        tt_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=cache_position,
        position_index=position,
    )
    output = ttnn.to_torch(tt_output).squeeze(0)
    tt_hidden.deallocate(True)
    cache_position.deallocate(True)
    tt_output.deallocate(True)
    return output


def _run_decode_query(model, device, hidden_states, position: int):
    tt_hidden = _tt_input(hidden_states, device)
    residual = ttnn.reshape(tt_hidden, (1, 1, EMITTED_BATCH, model.hidden_size))
    query, key, value = model._decode_qkv(residual, position_index=position)
    output = ttnn.to_torch(query).squeeze(0)
    query.deallocate(True)
    key.deallocate(True)
    value.deallocate(True)
    residual.deallocate(False)
    tt_hidden.deallocate(True)
    return output


def test_runtime_forwards_have_no_host_fallback():
    runtime_methods = (
        FunctionalDecoder.prefill_forward,
        FunctionalDecoder.decode_forward,
        FunctionalDecoder._prefill_attention,
        FunctionalDecoder._decode_attention,
        FunctionalDecoder._decode_qkv,
        FunctionalDecoder._mlp,
        FunctionalDecoder._rotary_slice,
        FunctionalDecoder._unpad_prefill_sequence,
        FunctionalDecoder._prepare_decode_heads,
        FunctionalDecoder._decode_attention_mask,
        FunctionalDecoder._validate_hidden,
        FunctionalDecoder._validate_caches,
        FunctionalDecoder._validate_cache_position,
    )
    forbidden = ("torch", "from_torch", "to_torch")
    for method in runtime_methods:
        source = inspect.getsource(method)
        for token in forbidden:
            assert token not in source, f"{method.__name__} contains forbidden runtime token {token!r}"


@pytest.mark.use_module_device
@pytest.mark.timeout(1800)
def test_synthetic_prefill_small_and_larger_and_decode(device):
    config = _config()
    state_dict = _synthetic_state_dict(config, IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    model = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=device,
    )

    generator = torch.Generator().manual_seed(17)
    for seq_len in (EMITTED_PREFILL_SEQUENCE, EMITTED_CACHE_LENGTH):
        hidden = torch.randn((EMITTED_BATCH, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
        expected = _hf_prefill(hf_layer, config, hidden)
        actual, key_cache, value_cache = _run_prefill(model, device, hidden)
        _assert_pcc(f"synthetic prefill seq={seq_len}", expected, actual, SYNTHETIC_PCC)
        key_cache.deallocate(True)
        value_cache.deallocate(True)

    past_len = EMITTED_PREFILL_SEQUENCE
    past_hidden = torch.randn((EMITTED_BATCH, past_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    decode_hidden = torch.randn((EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    hf_cache = DynamicCache(config=config)
    _hf_prefill(hf_layer, config, past_hidden, cache=hf_cache)
    assert hf_cache.get_seq_length(IR_REPRESENTATIVE_LAYER) == past_len
    _, key_cache, value_cache = _run_prefill(model, device, past_hidden)
    _assert_cache_prefix_matches_hf(hf_cache, key_cache, value_cache, past_len)
    expected_decode = _hf_decode(hf_layer, config, decode_hidden, hf_cache, past_len)
    expected_query = _hf_decode_query(hf_layer, config, decode_hidden, past_len)
    actual_query = _run_decode_query(model, device, decode_hidden, past_len)
    _assert_pcc("synthetic decode query", expected_query, actual_query, SYNTHETIC_PCC)
    actual_decode = _run_decode(model, device, decode_hidden, key_cache, value_cache, past_len)
    _assert_cache_prefix_matches_hf(hf_cache, key_cache, value_cache, past_len + 1)
    _assert_pcc("synthetic decode", expected_decode, actual_decode, SYNTHETIC_PCC)
    key_cache.deallocate(True)
    value_cache.deallocate(True)
    _release_model(model)


@pytest.mark.use_module_device
@pytest.mark.timeout(1800)
def test_real_weight_prefill_and_one_decode_step(device):
    config = _config()
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    model = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=device,
    )

    generator = torch.Generator().manual_seed(314)
    seq_len = EMITTED_PREFILL_SEQUENCE
    hidden = torch.randn((EMITTED_BATCH, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    hf_cache = DynamicCache(config=config)
    expected_prefill = _hf_prefill(hf_layer, config, hidden, cache=hf_cache)
    assert hf_cache.get_seq_length(IR_REPRESENTATIVE_LAYER) == seq_len
    actual_prefill, key_cache, value_cache = _run_prefill(model, device, hidden)
    _assert_pcc("real-weight prefill seq=17", expected_prefill, actual_prefill, REAL_WEIGHT_PCC)

    decode_hidden = torch.randn((EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    expected_decode = _hf_decode(hf_layer, config, decode_hidden, hf_cache, seq_len)
    actual_decode = _run_decode(model, device, decode_hidden, key_cache, value_cache, seq_len)
    _assert_pcc("real-weight decode position=17", expected_decode, actual_decode, REAL_WEIGHT_PCC)
    key_cache.deallocate(True)
    value_cache.deallocate(True)
    _release_model(model)
