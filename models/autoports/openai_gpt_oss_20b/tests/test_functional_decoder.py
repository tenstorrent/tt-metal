# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer, GptOssRotaryEmbedding

import ttnn
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    REPRESENTATIVE_LAYER,
    SUPPORTED_CONTEXT,
    FunctionalDecoder,
)
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import comp_pcc

CONFIG_DIR = Path("models/demos/gpt_oss/configs/gpt-oss-20b")
LAYER_IDX = REPRESENTATIVE_LAYER


def _config():
    config = AutoConfig.from_pretrained(CONFIG_DIR, local_files_only=True)
    config._attn_implementation = "eager"
    return config


def _canonical_key(suffix: str) -> str:
    return f"model.layers.{LAYER_IDX}.{suffix}"


def _synthetic_state_dict(config):
    """Full-shape synthetic layer; zero experts keep the HF reference tractable."""
    generator = torch.Generator().manual_seed(20260715)
    hidden = config.hidden_size
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    inter = config.intermediate_size
    experts = config.num_local_experts

    def randn(*shape, scale=0.01):
        return (torch.randn(*shape, generator=generator) * scale).to(torch.bfloat16)

    return {
        _canonical_key("input_layernorm.weight"): torch.ones(hidden, dtype=torch.bfloat16),
        _canonical_key("post_attention_layernorm.weight"): torch.ones(hidden, dtype=torch.bfloat16),
        _canonical_key("self_attn.q_proj.weight"): randn(q_dim, hidden),
        _canonical_key("self_attn.q_proj.bias"): randn(q_dim),
        _canonical_key("self_attn.k_proj.weight"): randn(kv_dim, hidden),
        _canonical_key("self_attn.k_proj.bias"): randn(kv_dim),
        _canonical_key("self_attn.v_proj.weight"): randn(kv_dim, hidden),
        _canonical_key("self_attn.v_proj.bias"): randn(kv_dim),
        _canonical_key("self_attn.o_proj.weight"): randn(hidden, q_dim),
        _canonical_key("self_attn.o_proj.bias"): randn(hidden),
        _canonical_key("self_attn.sinks"): randn(config.num_attention_heads),
        _canonical_key("mlp.router.weight"): torch.zeros(experts, hidden, dtype=torch.bfloat16),
        _canonical_key("mlp.router.bias"): torch.arange(experts, dtype=torch.bfloat16),
        _canonical_key("mlp.experts.gate_up_proj"): torch.zeros(experts, hidden, 2 * inter, dtype=torch.bfloat16),
        _canonical_key("mlp.experts.gate_up_proj_bias"): torch.zeros(experts, 2 * inter, dtype=torch.bfloat16),
        _canonical_key("mlp.experts.down_proj"): torch.zeros(experts, inter, hidden, dtype=torch.bfloat16),
        _canonical_key("mlp.experts.down_proj_bias"): torch.zeros(experts, hidden, dtype=torch.bfloat16),
    }


def _real_state_dict():
    explicit = os.environ.get("GPT_OSS_20B_REAL_WEIGHT_SHARD")
    roots = []
    if explicit:
        candidates = [Path(explicit)]
    else:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            roots.extend((Path(hf_home), Path(hf_home) / "hub"))
        candidates = [
            shard
            for root in roots
            for shard in root.glob("models--openai--gpt-oss-20b/snapshots/*/model-00000-of-00002.safetensors")
        ]
    weight_shard = next((candidate for candidate in candidates if candidate.is_file()), None)
    if weight_shard is None:
        pytest.skip("official GPT-OSS-20B layer-12 shard not found; set HF_HOME or " "GPT_OSS_20B_REAL_WEIGHT_SHARD")
    prefix = f"model.layers.{LAYER_IDX}."
    with safe_open(weight_shard, framework="pt", device="cpu") as handle:
        state = {key: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)}
    for projection in ("gate_up_proj", "down_proj"):
        blocks = state.pop(_canonical_key(f"mlp.experts.{projection}_blocks"))
        scales = state.pop(_canonical_key(f"mlp.experts.{projection}_scales"))
        state[_canonical_key(f"mlp.experts.{projection}")] = convert_moe_packed_tensors(blocks, scales)
    return state


def _hf_layer(config, state_dict):
    prefix = f"model.layers.{LAYER_IDX}."
    local_state = {key.removeprefix(prefix): value for key, value in state_dict.items()}
    with torch.device("meta"):
        layer = GptOssDecoderLayer(config, LAYER_IDX)
    missing, unexpected = layer.load_state_dict(local_state, strict=True, assign=True)
    assert not missing and not unexpected
    return layer.eval()


def _causal_mask(query_positions, key_positions, dtype, sliding_window=None):
    keys = key_positions.reshape(1, 1, 1, -1)
    queries = query_positions.reshape(1, 1, -1, 1)
    allowed = keys <= queries
    if sliding_window is not None:
        allowed &= keys > queries - sliding_window
    mask = torch.full(allowed.shape, torch.finfo(dtype).min, dtype=dtype)
    return mask.masked_fill(allowed, 0)


@torch.no_grad()
def _hf_forward(layer, rotary, hidden_states, positions, cache, *, return_post_attention=False):
    cos, sin = rotary(hidden_states, positions.unsqueeze(0))
    cached_keys = getattr(cache.layers[LAYER_IDX], "keys", None)
    stored_key_count = 0 if cached_keys is None else cached_keys.shape[-2]
    key_count = stored_key_count + hidden_states.shape[1]
    if stored_key_count and layer.self_attn.sliding_window is not None:
        key_count = min(key_count, layer.self_attn.sliding_window)
    key_positions = torch.arange(positions[-1] - key_count + 1, positions[-1] + 1)
    mask = _causal_mask(
        positions,
        key_positions,
        hidden_states.dtype,
        sliding_window=layer.self_attn.sliding_window,
    )
    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    attention_output, _ = layer.self_attn(
        hidden_states=normalized,
        attention_mask=mask,
        position_ids=positions.unsqueeze(0),
        position_embeddings=(cos, sin),
        past_key_values=cache,
        use_cache=True,
    )
    post_attention = residual + attention_output
    normalized = layer.post_attention_layernorm(post_attention)
    moe_output, _ = layer.mlp(normalized)
    output = post_attention + moe_output
    if return_post_attention:
        return output, post_attention
    return output


def _to_tt(hidden_states, mesh_device):
    return ttnn.from_torch(
        hidden_states.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_torch_raw(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def _to_torch(tensor):
    return _to_torch_raw(tensor).squeeze(0)


def _position_tensor(position, mesh_device):
    return ttnn.from_torch(
        torch.tensor([position], dtype=torch.int32),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _decode_mask(position, config, cache_length, mesh_device):
    key_positions = torch.arange(cache_length)
    mask = _causal_mask(
        torch.tensor([position]),
        key_positions,
        torch.bfloat16,
        sliding_window=config.sliding_window,
    )
    mask = mask.expand(1, 1, config.num_attention_heads, cache_length).contiguous()
    return ttnn.from_torch(
        mask,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _assert_pcc(name, reference, actual, threshold):
    passed, message = comp_pcc(reference.float(), actual.float(), threshold)
    print(f"PCC_RESULT path={name} threshold={threshold} {message}")
    assert passed, message


def test_runtime_contract_and_no_host_fallback():
    assert issubclass(FunctionalDecoder, LightweightModule)
    assert EMITTED_BATCH == 1
    assert EMITTED_PREFILL_SEQUENCE == 17
    assert EMITTED_CACHE_LENGTH == 128
    assert SUPPORTED_CONTEXT == 21_248
    assert "NotImplementedError" not in inspect.getsource(FunctionalDecoder.prefill_forward)
    assert "NotImplementedError" not in inspect.getsource(FunctionalDecoder.decode_forward)

    runtime_methods = (
        FunctionalDecoder._validate_hidden_states,
        FunctionalDecoder._prefill_attention,
        FunctionalDecoder._decode_attention,
        FunctionalDecoder._moe_forward,
        FunctionalDecoder.prefill_forward,
        FunctionalDecoder.decode_forward,
        FunctionalDecoder.forward,
    )
    forbidden = (
        "torch",
        "from_torch",
        "to_torch",
        "from_device",
        "to_device",
        ".cpu(",
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "mesh_partition",
        "to_layout",
        "to_memory_config",
        "reshard",
        "tilize",
        "untilize",
    )
    for method in runtime_methods:
        source = inspect.getsource(method)
        assert all(token not in source for token in forbidden), method.__name__


def test_supported_context_bound_is_enforced_before_weight_loading(expect_error):
    config = _config()
    with expect_error(ValueError, "max_cache_len must be"):
        FunctionalDecoder.from_state_dict(
            {},
            hf_config=config,
            layer_idx=LAYER_IDX,
            mesh_device=object(),
            max_cache_len=SUPPORTED_CONTEXT + 1,
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [EMITTED_PREFILL_SEQUENCE, EMITTED_CACHE_LENGTH, 256])
def test_synthetic_prefill_matches_hf(mesh_device, seq_len):
    config = _config()
    state = _synthetic_state_dict(config)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=seq_len,
    )
    hf_layer = _hf_layer(config, state)
    rotary = GptOssRotaryEmbedding(config)
    generator = torch.Generator().manual_seed(1000 + seq_len)
    hidden = torch.randn(1, seq_len, config.hidden_size, generator=generator).to(torch.bfloat16)

    hf_cache = DynamicCache(config=config)
    reference = _hf_forward(hf_layer, rotary, hidden, torch.arange(seq_len), hf_cache)
    key_cache, value_cache = decoder.create_kv_cache()
    actual = decoder.prefill_forward(
        _to_tt(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
    )
    _assert_pcc(f"synthetic-prefill-{seq_len}", reference, _to_torch(actual), 0.99)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_real_weight_prefill_and_decode_match_hf(mesh_device):
    config = _config()
    state = _real_state_dict()
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=288,
    )
    hf_layer = _hf_layer(config, state)
    rotary = GptOssRotaryEmbedding(config)
    generator = torch.Generator().manual_seed(4242)

    prefill_len = EMITTED_PREFILL_SEQUENCE
    prefill_hidden = torch.randn(1, prefill_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    hf_cache = DynamicCache(config=config)
    prefill_reference = _hf_forward(hf_layer, rotary, prefill_hidden, torch.arange(prefill_len), hf_cache)
    key_cache, value_cache = decoder.create_kv_cache()
    prefill_actual = decoder.prefill_forward(
        _to_tt(prefill_hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
    )
    _assert_pcc("real-prefill-17", prefill_reference, _to_torch(prefill_actual), 0.99)

    decode_hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    decode_reference = _hf_forward(hf_layer, rotary, decode_hidden, torch.tensor([prefill_len]), hf_cache)
    decode_actual = decoder.decode_forward(
        _to_tt(decode_hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=prefill_len,
        cache_position_tensor=_position_tensor(prefill_len, mesh_device),
        attention_mask=_decode_mask(prefill_len, config, decoder.max_cache_len, mesh_device),
    )
    _assert_pcc("real-decode-position-17", decode_reference, _to_torch(decode_actual), 0.99)

    # Exercise decode with stale keys outside layer 12's 128-token sliding
    # window. At position 256, only positions 129..256 are eligible.
    boundary_prefix_len = 256
    boundary_hidden = torch.randn(1, boundary_prefix_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    boundary_hf_cache = DynamicCache(config=config)
    boundary_reference = _hf_forward(
        hf_layer,
        rotary,
        boundary_hidden,
        torch.arange(boundary_prefix_len),
        boundary_hf_cache,
    )
    boundary_key_cache, boundary_value_cache = decoder.create_kv_cache()
    boundary_actual = decoder.prefill_forward(
        _to_tt(boundary_hidden, mesh_device),
        key_cache=boundary_key_cache,
        value_cache=boundary_value_cache,
    )
    _assert_pcc(
        "real-prefill-256-sliding-primer",
        boundary_reference,
        _to_torch(boundary_actual),
        0.99,
    )
    boundary_tt_key_cache = _to_torch_raw(boundary_key_cache)
    boundary_tt_value_cache = _to_torch_raw(boundary_value_cache)
    _assert_pcc(
        "real-prefill-256-sliding-key-cache",
        boundary_hf_cache.layers[LAYER_IDX].keys,
        boundary_tt_key_cache[:, :, 129:256, :],
        0.99,
    )
    _assert_pcc(
        "real-prefill-256-sliding-value-cache",
        boundary_hf_cache.layers[LAYER_IDX].values,
        boundary_tt_value_cache[:, :, 129:256, :],
        0.99,
    )

    boundary_decode_hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    boundary_decode_reference, boundary_post_attention_reference = _hf_forward(
        hf_layer,
        rotary,
        boundary_decode_hidden,
        torch.tensor([boundary_prefix_len]),
        boundary_hf_cache,
        return_post_attention=True,
    )
    boundary_decode_tt_input = _to_tt(boundary_decode_hidden, mesh_device)
    boundary_post_attention_actual = decoder._decode_attention(
        boundary_decode_tt_input,
        boundary_key_cache,
        boundary_value_cache,
        boundary_prefix_len,
        _position_tensor(boundary_prefix_len, mesh_device),
        _decode_mask(boundary_prefix_len, config, decoder.max_cache_len, mesh_device),
    )
    _assert_pcc(
        "real-decode-position-256-key-cache-update",
        boundary_hf_cache.layers[LAYER_IDX].keys[:, :, -1:, :],
        _to_torch_raw(boundary_key_cache)[:, :, boundary_prefix_len : boundary_prefix_len + 1, :],
        0.99,
    )
    _assert_pcc(
        "real-decode-position-256-value-cache-update",
        boundary_hf_cache.layers[LAYER_IDX].values[:, :, -1:, :],
        _to_torch_raw(boundary_value_cache)[:, :, boundary_prefix_len : boundary_prefix_len + 1, :],
        0.99,
    )
    _assert_pcc(
        "real-decode-position-256-sliding-attention",
        boundary_post_attention_reference,
        _to_torch(boundary_post_attention_actual),
        0.99,
    )
    boundary_decode_actual = decoder._moe_forward(
        boundary_post_attention_actual,
        1,
    )
    _assert_pcc(
        "real-decode-position-256-sliding-window",
        boundary_decode_reference,
        _to_torch(boundary_decode_actual),
        0.99,
    )
