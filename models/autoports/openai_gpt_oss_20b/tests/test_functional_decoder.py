# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import inspect
import json
from pathlib import Path

import pytest
import torch
from huggingface_hub import try_to_load_from_cache
from safetensors import safe_open
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer, GptOssRotaryEmbedding

import ttnn
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    REPRESENTATIVE_LAYER,
    FunctionalDecoder,
)
from models.common.utility_functions import comp_pcc

MODEL_ID = "openai/gpt-oss-20b"
LAYER_IDX = REPRESENTATIVE_LAYER
DENSE_SUFFIXES = (
    "self_attn.sinks",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "self_attn.o_proj.bias",
    "mlp.router.weight",
    "mlp.router.bias",
    "mlp.experts.gate_up_proj",
    "mlp.experts.gate_up_proj_bias",
    "mlp.experts.down_proj",
    "mlp.experts.down_proj_bias",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)
RAW_EXPERT_SUFFIXES = (
    "mlp.experts.gate_up_proj_blocks",
    "mlp.experts.gate_up_proj_scales",
    "mlp.experts.down_proj_blocks",
    "mlp.experts.down_proj_scales",
)


def _config():
    config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    config._attn_implementation = "eager"
    return config


def _weight_key(suffix: str) -> str:
    return f"model.layers.{LAYER_IDX}.{suffix}"


def _normal(shape, generator, scale=0.02):
    tensor = torch.empty(shape, dtype=torch.bfloat16)
    return tensor.normal_(mean=0.0, std=scale, generator=generator)


def _synthetic_state(config) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    hidden = config.hidden_size
    head_dim = config.head_dim
    q_width = config.num_attention_heads * head_dim
    kv_width = config.num_key_value_heads * head_dim
    experts = config.num_local_experts
    intermediate = config.intermediate_size

    # Dense expert allocations match the real interface. Zero projections plus
    # nonzero biases keep this deterministic while exercising routing/scatter.
    gate_up = torch.zeros((experts, hidden, 2 * intermediate), dtype=torch.bfloat16)
    down = torch.zeros((experts, intermediate, hidden), dtype=torch.bfloat16)
    router_bias = torch.linspace(-0.5, 0.5, experts, dtype=torch.bfloat16)
    return {
        _weight_key("self_attn.sinks"): _normal((config.num_attention_heads,), generator, scale=0.1),
        _weight_key("self_attn.q_proj.weight"): _normal((q_width, hidden), generator),
        _weight_key("self_attn.q_proj.bias"): _normal((q_width,), generator),
        _weight_key("self_attn.k_proj.weight"): _normal((kv_width, hidden), generator),
        _weight_key("self_attn.k_proj.bias"): _normal((kv_width,), generator),
        _weight_key("self_attn.v_proj.weight"): _normal((kv_width, hidden), generator),
        _weight_key("self_attn.v_proj.bias"): _normal((kv_width,), generator),
        _weight_key("self_attn.o_proj.weight"): _normal((hidden, q_width), generator),
        _weight_key("self_attn.o_proj.bias"): _normal((hidden,), generator),
        _weight_key("mlp.router.weight"): _normal((experts, hidden), generator),
        _weight_key("mlp.router.bias"): router_bias,
        _weight_key("mlp.experts.gate_up_proj"): gate_up,
        _weight_key("mlp.experts.gate_up_proj_bias"): _normal((experts, 2 * intermediate), generator),
        _weight_key("mlp.experts.down_proj"): down,
        _weight_key("mlp.experts.down_proj_bias"): _normal((experts, hidden), generator),
        _weight_key("input_layernorm.weight"): 1.0 + _normal((hidden,), generator, scale=0.01),
        _weight_key("post_attention_layernorm.weight"): 1.0 + _normal((hidden,), generator, scale=0.01),
    }


def _real_state() -> dict[str, torch.Tensor]:
    index_path = try_to_load_from_cache(MODEL_ID, "model.safetensors.index.json")
    if not isinstance(index_path, str) or not Path(index_path).is_file():
        pytest.skip(f"{MODEL_ID} safetensors are not present in the local Hugging Face cache")
    index = json.loads(Path(index_path).read_text())
    weight_map = index["weight_map"]

    requested = [suffix for suffix in DENSE_SUFFIXES if "gate_up_proj" not in suffix and "down_proj" not in suffix]
    requested += [
        "mlp.experts.gate_up_proj_bias",
        "mlp.experts.down_proj_bias",
        *RAW_EXPERT_SUFFIXES,
    ]
    by_shard: dict[str, list[str]] = {}
    for suffix in requested:
        key = _weight_key(suffix)
        if key not in weight_map:
            pytest.fail(f"safetensors index does not contain {key}")
        by_shard.setdefault(weight_map[key], []).append(key)

    state: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        shard_path = try_to_load_from_cache(MODEL_ID, shard)
        if not isinstance(shard_path, str) or not Path(shard_path).is_file():
            pytest.skip(f"local Hugging Face cache is missing {shard}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in keys:
                state[key] = handle.get_tensor(key)

    for projection in ("gate_up_proj", "down_proj"):
        blocks_key = _weight_key(f"mlp.experts.{projection}_blocks")
        scales_key = _weight_key(f"mlp.experts.{projection}_scales")
        state[_weight_key(f"mlp.experts.{projection}")] = convert_moe_packed_tensors(
            state.pop(blocks_key),
            state.pop(scales_key),
            dtype=torch.bfloat16,
        )
    for key, value in tuple(state.items()):
        state[key] = value.to(torch.bfloat16)
    return state


def _hf_layer(state: dict[str, torch.Tensor], config) -> GptOssDecoderLayer:
    with torch.device("meta"):
        layer = GptOssDecoderLayer(config, layer_idx=LAYER_IDX)
    layer_state = {suffix: state[_weight_key(suffix)] for suffix in DENSE_SUFFIXES}
    layer.load_state_dict(layer_state, strict=True, assign=True)
    return layer.eval()


@torch.no_grad()
def _reference_layer(
    layer: GptOssDecoderLayer,
    hidden_states: torch.Tensor,
    config,
    *,
    start_pos: int = 0,
    cache: DynamicCache | None = None,
):
    layer_input = hidden_states[0]
    batch, seq_len, _ = layer_input.shape
    positions = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long).unsqueeze(0)
    rotary = GptOssRotaryEmbedding(config)
    cos, sin = rotary(layer_input, positions)

    query_positions = torch.arange(start_pos, start_pos + seq_len).view(seq_len, 1)
    key_positions = torch.arange(start_pos + seq_len).view(1, start_pos + seq_len)
    invalid = key_positions > query_positions
    if layer.self_attn.sliding_window is not None:
        invalid = invalid | (key_positions <= query_positions - layer.self_attn.sliding_window)
    attention_mask = torch.zeros((batch, 1, seq_len, start_pos + seq_len), dtype=layer_input.dtype)
    attention_mask.masked_fill_(invalid.view(1, 1, seq_len, start_pos + seq_len), torch.finfo(layer_input.dtype).min)

    if cache is None:
        cache = DynamicCache(config=config)
    output = layer(
        layer_input,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        position_embeddings=(cos, sin),
    )
    current_key = cache.layers[LAYER_IDX].keys[:, :, -seq_len:, :]
    current_value = cache.layers[LAYER_IDX].values[:, :, -seq_len:, :]
    return output.unsqueeze(0), current_key, current_value, cache


def _tt_tensor(tensor: torch.Tensor, mesh_device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _empty_caches(config, mesh_device, *, batch=EMITTED_BATCH):
    shape = (batch, config.num_key_value_heads, EMITTED_CACHE_LENGTH, config.head_dim)
    key = _tt_tensor(torch.zeros(shape, dtype=torch.bfloat16), mesh_device)
    value = _tt_tensor(torch.zeros(shape, dtype=torch.bfloat16), mesh_device)
    return key, value


def _to_host(tensor):
    result = ttnn.to_torch(tensor)
    if isinstance(result, list):
        if len(result) != 1:
            raise AssertionError(f"expected one tensor from the 1x1 mesh, got {len(result)}")
        result = result[0]
    return result


def _assert_pcc(reference, actual, threshold: float, label: str):
    passed, message = comp_pcc(reference.float(), actual.float(), pcc=threshold)
    print(f"{label}: {message}")
    assert passed, f"{label}: {message}"


def test_runtime_forwards_have_no_host_or_collective_fallback():
    forbidden = (
        "torch",
        "from_torch",
        "to_torch",
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "mesh_partition",
    )
    runtime_methods = (
        FunctionalDecoder._moe_forward,
        FunctionalDecoder.prefill_forward,
        FunctionalDecoder.decode_forward,
    )
    for method in runtime_methods:
        source = inspect.getsource(method)
        for token in forbidden:
            assert token not in source, f"{method.__name__} contains forbidden runtime token {token!r}"


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_synthetic_prefill_pcc_small_and_emitted_sequence(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
    )
    reference_layer = _hf_layer(state, config)

    for seq_len in (4, EMITTED_PREFILL_SEQUENCE):
        generator = torch.Generator().manual_seed(100 + seq_len)
        hidden = torch.randn((1, EMITTED_BATCH, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
        reference, reference_key, reference_value, _ = _reference_layer(reference_layer, hidden, config)
        key_cache, value_cache = _empty_caches(config, mesh_device)
        actual = decoder.prefill_forward(_tt_tensor(hidden, mesh_device), key_cache, value_cache)

        _assert_pcc(reference, _to_host(actual), 0.99, f"synthetic prefill seq={seq_len} output")
        _assert_pcc(reference_key, _to_host(key_cache)[:, :, :seq_len, :], 0.99, f"synthetic seq={seq_len} key")
        _assert_pcc(reference_value, _to_host(value_cache)[:, :, :seq_len, :], 0.99, f"synthetic seq={seq_len} value")

    del decoder, reference_layer, state
    gc.collect()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_synthetic_non_emitted_batch_prefill_and_decode(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    batch = 2
    seq_len = 4
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
    )
    reference_layer = _hf_layer(state, config)
    key_cache, value_cache = _empty_caches(config, mesh_device, batch=batch)

    generator = torch.Generator().manual_seed(204)
    prefill_hidden = torch.randn((1, batch, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    reference_prefill, reference_key, reference_value, reference_cache = _reference_layer(
        reference_layer, prefill_hidden, config
    )
    actual_prefill = decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache)
    _assert_pcc(reference_prefill, _to_host(actual_prefill), 0.99, "synthetic batch=2 prefill output")
    _assert_pcc(reference_key, _to_host(key_cache)[:, :, :seq_len, :], 0.99, "synthetic batch=2 prefill key")
    _assert_pcc(
        reference_value,
        _to_host(value_cache)[:, :, :seq_len, :],
        0.99,
        "synthetic batch=2 prefill value",
    )

    decode_hidden = torch.randn((1, batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    reference_decode, decode_key, decode_value, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=seq_len,
        cache=reference_cache,
    )
    actual_decode = decoder.decode_forward(
        _tt_tensor(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos=seq_len,
    )
    _assert_pcc(reference_decode, _to_host(actual_decode), 0.99, "synthetic batch=2 decode output")
    cache_slice = slice(seq_len, seq_len + 1)
    _assert_pcc(decode_key, _to_host(key_cache)[:, :, cache_slice, :], 0.99, "synthetic batch=2 decode key")
    _assert_pcc(
        decode_value,
        _to_host(value_cache)[:, :, cache_slice, :],
        0.99,
        "synthetic batch=2 decode value",
    )

    del decoder, reference_layer, state
    gc.collect()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_real_weight_prefill_and_decode_pcc(mesh_device):
    config = _config()
    state = _real_state()
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
    )
    reference_layer = _hf_layer(state, config)
    key_cache, value_cache = _empty_caches(config, mesh_device)

    # Fixed seed chosen for a non-degenerate top-k routing margin. Near-tied
    # router logits can legitimately switch experts after BF16 device rounding.
    generator = torch.Generator().manual_seed(9090)
    prefill_hidden = torch.randn(
        (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
    ).to(torch.bfloat16)
    reference_prefill, reference_key, reference_value, reference_cache = _reference_layer(
        reference_layer, prefill_hidden, config
    )
    actual_prefill = decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache)
    _assert_pcc(reference_prefill, _to_host(actual_prefill), 0.99, "real prefill seq=17 output")
    _assert_pcc(reference_key, _to_host(key_cache)[:, :, :EMITTED_PREFILL_SEQUENCE, :], 0.99, "real prefill key")
    _assert_pcc(
        reference_value,
        _to_host(value_cache)[:, :, :EMITTED_PREFILL_SEQUENCE, :],
        0.99,
        "real prefill value",
    )

    decode_hidden = torch.randn((1, EMITTED_BATCH, 1, config.hidden_size), generator=generator).to(torch.bfloat16)
    reference_decode, decode_key, decode_value, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    actual_decode = decoder.decode_forward(
        _tt_tensor(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    _assert_pcc(reference_decode, _to_host(actual_decode), 0.99, "real decode pos=17 output")
    cache_slice = slice(EMITTED_PREFILL_SEQUENCE, EMITTED_PREFILL_SEQUENCE + 1)
    _assert_pcc(decode_key, _to_host(key_cache)[:, :, cache_slice, :], 0.99, "real decode key append")
    _assert_pcc(decode_value, _to_host(value_cache)[:, :, cache_slice, :], 0.99, "real decode value append")

    del decoder, reference_layer, state
    gc.collect()
