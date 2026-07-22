# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_32b.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    HF_MODEL,
    REPRESENTATIVE_LAYER,
    FunctionalDecoder,
)
from models.common.utility_functions import comp_pcc

REAL_WEIGHT_DIR_ENV = "QWEN3_32B_REAL_WEIGHT_DIR"
WEIGHT_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _config():
    config = AutoConfig.from_pretrained(HF_MODEL, local_files_only=True)
    config._attn_implementation = "eager"
    return config


def _weight_key(suffix: str) -> str:
    return f"model.layers.{REPRESENTATIVE_LAYER}.{suffix}"


def _synthetic_state(config) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(32017)
    hidden = config.hidden_size
    head_dim = config.head_dim
    attention_width = config.num_attention_heads * head_dim
    kv_width = config.num_key_value_heads * head_dim
    intermediate = config.intermediate_size

    def normal(shape, scale=0.01):
        tensor = torch.empty(shape, dtype=torch.bfloat16)
        return tensor.normal_(mean=0.0, std=scale, generator=generator)

    return {
        _weight_key("input_layernorm.weight"): 1.0 + normal((hidden,), scale=0.01),
        _weight_key("post_attention_layernorm.weight"): 1.0 + normal((hidden,), scale=0.01),
        _weight_key("self_attn.q_proj.weight"): normal((attention_width, hidden)),
        _weight_key("self_attn.k_proj.weight"): normal((kv_width, hidden)),
        _weight_key("self_attn.v_proj.weight"): normal((kv_width, hidden)),
        _weight_key("self_attn.o_proj.weight"): normal((hidden, attention_width)),
        _weight_key("self_attn.q_norm.weight"): 1.0 + normal((head_dim,), scale=0.01),
        _weight_key("self_attn.k_norm.weight"): 1.0 + normal((head_dim,), scale=0.01),
        _weight_key("mlp.gate_proj.weight"): normal((intermediate, hidden)),
        _weight_key("mlp.up_proj.weight"): normal((intermediate, hidden)),
        _weight_key("mlp.down_proj.weight"): normal((hidden, intermediate)),
    }


def _real_state() -> dict[str, torch.Tensor]:
    directory_text = os.environ.get(REAL_WEIGHT_DIR_ENV)
    if not directory_text:
        pytest.skip(f"Set {REAL_WEIGHT_DIR_ENV} to a local Qwen3-32B HF snapshot")
    directory = Path(directory_text)
    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        pytest.fail(f"{REAL_WEIGHT_DIR_ENV} does not contain model.safetensors.index.json: {directory}")

    weight_map = json.loads(index_path.read_text())["weight_map"]
    keys_by_shard: dict[str, list[str]] = {}
    for suffix in WEIGHT_SUFFIXES:
        key = _weight_key(suffix)
        shard = weight_map.get(key)
        if shard is None:
            pytest.fail(f"Checkpoint index does not contain {key}")
        keys_by_shard.setdefault(shard, []).append(key)

    state = {}
    for shard_name, keys in keys_by_shard.items():
        shard_path = directory / shard_name
        if not shard_path.is_file():
            pytest.fail(f"Checkpoint shard required for layer {REPRESENTATIVE_LAYER} is missing: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for key in keys:
                if key not in available:
                    pytest.fail(f"{shard_path} does not contain {key}")
                state[key] = handle.get_tensor(key).to(torch.bfloat16)
    return state


def _hf_layer(state: dict[str, torch.Tensor], config) -> Qwen3DecoderLayer:
    layer = Qwen3DecoderLayer(config, layer_idx=REPRESENTATIVE_LAYER).to(dtype=torch.bfloat16).eval()
    layer_state = {suffix: state[_weight_key(suffix)] for suffix in WEIGHT_SUFFIXES}
    layer.load_state_dict(layer_state, strict=True)
    return layer


@torch.no_grad()
def _reference_layer(
    layer: Qwen3DecoderLayer,
    hidden_states: torch.Tensor,
    config,
    *,
    start_pos: int = 0,
    cache: DynamicCache | None = None,
):
    layer_input = hidden_states[0]
    batch, seq_len, _ = layer_input.shape
    positions = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long).unsqueeze(0)
    rotary = Qwen3RotaryEmbedding(config)
    cos, sin = rotary(layer_input, positions)

    absolute_query = torch.arange(start_pos, start_pos + seq_len).view(seq_len, 1)
    absolute_key = torch.arange(start_pos + seq_len).view(1, start_pos + seq_len)
    attention_mask = torch.zeros((batch, 1, seq_len, start_pos + seq_len), dtype=layer_input.dtype)
    attention_mask.masked_fill_(absolute_key > absolute_query, torch.finfo(layer_input.dtype).min)

    if cache is None:
        cache = DynamicCache(config=config)
    output = layer(
        layer_input,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        position_embeddings=(cos, sin),
    )
    current_key = cache.layers[REPRESENTATIVE_LAYER].keys[:, :, -seq_len:, :]
    current_value = cache.layers[REPRESENTATIVE_LAYER].values[:, :, -seq_len:, :]
    return output.unsqueeze(0), current_key, current_value, cache


def _tt_tensor(tensor: torch.Tensor, mesh_device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _empty_caches(config, mesh_device, *, batch=EMITTED_BATCH, max_cache_len=EMITTED_CACHE_LENGTH):
    shape = (batch, config.num_key_value_heads, max_cache_len, config.head_dim)
    key = _tt_tensor(torch.zeros(shape, dtype=torch.bfloat16), mesh_device)
    value = _tt_tensor(torch.zeros(shape, dtype=torch.bfloat16), mesh_device)
    return key, value


def _to_host(tensor):
    result = ttnn.to_torch(tensor)
    if isinstance(result, list):
        if len(result) != 1:
            raise AssertionError(f"Expected one tensor from the 1x1 mesh, got {len(result)}")
        result = result[0]
    return result


def _assert_pcc(reference, actual, threshold: float, label: str) -> float:
    passed, pcc = comp_pcc(reference.float(), actual.float(), pcc=threshold)
    print(f"{label}: {pcc}")
    assert passed, f"{label}: PCC {pcc} is below {threshold}"
    return float(pcc)


def test_runtime_forwards_have_no_host_fallback():
    forbidden = ("torch", "from_torch", "to_torch", "numpy", ".cpu(")
    methods = (
        FunctionalDecoder._mlp_forward,
        FunctionalDecoder.prefill_forward,
        FunctionalDecoder.decode_forward,
    )
    for method in methods:
        source = inspect.getsource(method)
        for token in forbidden:
            assert token not in source, f"{method.__name__} contains forbidden runtime token {token!r}"


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_synthetic_prefill_and_decode_match_hf(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
    )
    reference_layer = _hf_layer(state, config)

    for seq_len in (4, EMITTED_PREFILL_SEQUENCE, EMITTED_CACHE_LENGTH):
        generator = torch.Generator().manual_seed(1000 + seq_len)
        hidden = torch.randn(
            (1, EMITTED_BATCH, seq_len, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        reference, reference_key, reference_value, _ = _reference_layer(reference_layer, hidden, config)
        key_cache, value_cache = _empty_caches(config, mesh_device)
        actual = decoder.prefill_forward(_tt_tensor(hidden, mesh_device), key_cache, value_cache)

        _assert_pcc(reference, _to_host(actual), 0.995, f"synthetic prefill seq={seq_len} output")
        _assert_pcc(
            reference_key,
            _to_host(key_cache)[:, :, :seq_len, :],
            0.99,
            f"synthetic prefill seq={seq_len} key cache",
        )
        _assert_pcc(
            reference_value,
            _to_host(value_cache)[:, :, :seq_len, :],
            0.99,
            f"synthetic prefill seq={seq_len} value cache",
        )

    generator = torch.Generator().manual_seed(32032)
    prefill_hidden = torch.randn(
        (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    _, _, _, reference_cache = _reference_layer(reference_layer, prefill_hidden, config)
    key_cache, value_cache = _empty_caches(config, mesh_device)
    decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache)

    decode_hidden = torch.randn(
        (1, EMITTED_BATCH, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference, reference_key, reference_value, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    actual = decoder.decode_forward(
        _tt_tensor(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    _assert_pcc(reference, _to_host(actual), 0.995, "synthetic decode output")
    _assert_pcc(
        reference_key,
        _to_host(key_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :],
        0.99,
        "synthetic decode key append",
    )
    _assert_pcc(
        reference_value,
        _to_host(value_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :],
        0.99,
        "synthetic decode value append",
    )


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_real_weight_prefill_and_decode_match_hf(mesh_device):
    config = _config()
    state = _real_state()
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
    )
    reference_layer = _hf_layer(state, config)
    key_cache, value_cache = _empty_caches(config, mesh_device)

    generator = torch.Generator().manual_seed(32)
    prefill_hidden = torch.randn(
        (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference, reference_key, reference_value, reference_cache = _reference_layer(
        reference_layer,
        prefill_hidden,
        config,
    )
    actual = decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache)
    _assert_pcc(reference, _to_host(actual), 0.995, "real prefill output")
    _assert_pcc(
        reference_key,
        _to_host(key_cache)[:, :, :EMITTED_PREFILL_SEQUENCE, :],
        0.99,
        "real prefill key cache",
    )
    _assert_pcc(
        reference_value,
        _to_host(value_cache)[:, :, :EMITTED_PREFILL_SEQUENCE, :],
        0.99,
        "real prefill value cache",
    )

    decode_hidden = torch.randn(
        (1, EMITTED_BATCH, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference, reference_key, reference_value, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    actual = decoder.decode_forward(
        _tt_tensor(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    _assert_pcc(reference, _to_host(actual), 0.995, "real decode output")
    _assert_pcc(
        reference_key,
        _to_host(key_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :],
        0.99,
        "real decode key append",
    )
    _assert_pcc(
        reference_value,
        _to_host(value_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :],
        0.99,
        "real decode value append",
    )
