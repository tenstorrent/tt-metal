# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

import ttnn
from models.autoports.qwen_qwen2_5_coder_32b_instruct.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen2_5_coder_32b_instruct.tt.optimized_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    HF_MODEL,
    REPRESENTATIVE_LAYER,
    OptimizationConfig,
    OptimizedDecoder,
)
from models.common.utility_functions import comp_pcc

REAL_WEIGHT_DIR_ENV = "QWEN2_5_CODER_32B_REAL_WEIGHT_DIR"
PROFILE_ENV = "QWEN2_5_CODER_32B_OPT_PROFILE"
PERF_ENV = "QWEN2_5_CODER_32B_OPT_PERF"
PERF_DECODER_ENV = "QWEN2_5_CODER_32B_PERF_DECODER"
PERF_BATCH_ENV = "QWEN2_5_CODER_32B_PERF_BATCH"
PERF_REPS_ENV = "QWEN2_5_CODER_32B_PERF_REPS"

# The completed functional decoder's real-weight layer-32 measurements are
# the stage gate, not a permissive generic transformer threshold.
FUNCTIONAL_PREFILL_PCC = 0.99878
FUNCTIONAL_DECODE_PCC = 0.99892
FUNCTIONAL_CACHE_PCC = 0.9998
SYNTHETIC_CACHE_PCC = 0.9995

WEIGHT_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _config():
    config = AutoConfig.from_pretrained(HF_MODEL, local_files_only=True)
    config._attn_implementation = "eager"
    return config


def _head_dim(config) -> int:
    return int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)


def _weight_key(layer_idx: int, suffix: str) -> str:
    return f"model.layers.{layer_idx}.{suffix}"


def _synthetic_state(config, layer_idx=REPRESENTATIVE_LAYER) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(25032 + layer_idx)
    hidden = config.hidden_size
    head_dim = _head_dim(config)
    attention_width = config.num_attention_heads * head_dim
    kv_width = config.num_key_value_heads * head_dim
    intermediate = config.intermediate_size

    def normal(shape, scale=0.01):
        tensor = torch.empty(shape, dtype=torch.bfloat16)
        return tensor.normal_(mean=0.0, std=scale, generator=generator)

    return {
        _weight_key(layer_idx, "input_layernorm.weight"): 1.0 + normal((hidden,), scale=0.01),
        _weight_key(layer_idx, "post_attention_layernorm.weight"): 1.0 + normal((hidden,), scale=0.01),
        _weight_key(layer_idx, "self_attn.q_proj.weight"): normal((attention_width, hidden)),
        _weight_key(layer_idx, "self_attn.q_proj.bias"): normal((attention_width,)),
        _weight_key(layer_idx, "self_attn.k_proj.weight"): normal((kv_width, hidden)),
        _weight_key(layer_idx, "self_attn.k_proj.bias"): normal((kv_width,)),
        _weight_key(layer_idx, "self_attn.v_proj.weight"): normal((kv_width, hidden)),
        _weight_key(layer_idx, "self_attn.v_proj.bias"): normal((kv_width,)),
        _weight_key(layer_idx, "self_attn.o_proj.weight"): normal((hidden, attention_width)),
        _weight_key(layer_idx, "mlp.gate_proj.weight"): normal((intermediate, hidden)),
        _weight_key(layer_idx, "mlp.up_proj.weight"): normal((intermediate, hidden)),
        _weight_key(layer_idx, "mlp.down_proj.weight"): normal((hidden, intermediate)),
    }


def _real_weight_dir() -> Path:
    directory_text = os.environ.get(REAL_WEIGHT_DIR_ENV)
    if not directory_text:
        pytest.skip(f"Set {REAL_WEIGHT_DIR_ENV} to a local Qwen2.5-Coder-32B HF snapshot")
    directory = Path(directory_text)
    if not (directory / "model.safetensors.index.json").is_file():
        pytest.fail(f"{REAL_WEIGHT_DIR_ENV} does not contain model.safetensors.index.json: {directory}")
    return directory


def _real_state(layer_idx: int) -> dict[str, torch.Tensor]:
    directory = _real_weight_dir()
    weight_map = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    keys_by_shard: dict[str, list[str]] = {}
    for suffix in WEIGHT_SUFFIXES:
        key = _weight_key(layer_idx, suffix)
        shard = weight_map.get(key)
        if shard is None:
            pytest.fail(f"Checkpoint index does not contain {key}")
        keys_by_shard.setdefault(shard, []).append(key)

    state = {}
    for shard_name, keys in keys_by_shard.items():
        shard_path = directory / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in keys:
                state[key] = handle.get_tensor(key).to(torch.bfloat16)
    return state


def _recorded_activation(seq_len: int, *, seed: int = 250320) -> torch.Tensor:
    """Stable nontrivial activation used with checkpoint decoder weights."""

    return torch.randn(
        (1, 1, seq_len, 5120),
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.bfloat16,
    )


def _hf_layer(state: dict[str, torch.Tensor], config, layer_idx: int) -> Qwen2DecoderLayer:
    layer = Qwen2DecoderLayer(config, layer_idx=layer_idx).to(dtype=torch.bfloat16).eval()
    layer_state = {suffix: state[_weight_key(layer_idx, suffix)] for suffix in WEIGHT_SUFFIXES}
    layer.load_state_dict(layer_state, strict=True)
    return layer


@torch.no_grad()
def _reference_layer(
    layer: Qwen2DecoderLayer,
    hidden_states: torch.Tensor,
    config,
    layer_idx: int,
    *,
    start_pos: int = 0,
    cache: DynamicCache | None = None,
):
    layer_input = hidden_states[0]
    batch, seq_len, _ = layer_input.shape
    positions = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long).unsqueeze(0)
    rotary = Qwen2RotaryEmbedding(config)
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
    current_key = cache.layers[layer_idx].keys[:, :, -seq_len:, :]
    current_value = cache.layers[layer_idx].values[:, :, -seq_len:, :]
    return output.unsqueeze(0), current_key, current_value, cache


def _tt_tensor(tensor: torch.Tensor, mesh_device, *, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=memory_config,
    )


def _empty_caches(config, mesh_device, *, batch: int, max_cache_len: int, dtype):
    shape = (batch, config.num_key_value_heads, max_cache_len, _head_dim(config))
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    return _tt_tensor(zeros, mesh_device, dtype=dtype), _tt_tensor(zeros, mesh_device, dtype=dtype)


def _to_host(tensor):
    result = ttnn.to_torch(tensor)
    if isinstance(result, list):
        assert len(result) == 1
        result = result[0]
    return result


def _assert_pcc(reference, actual, threshold: float, label: str) -> float:
    passed, pcc = comp_pcc(reference.float(), actual.float(), pcc=threshold)
    print(f"{label}: {pcc}")
    assert passed, f"{label}: PCC {pcc} is below {threshold}"
    return float(pcc)


def _profile_from_env(*, batch: int = 1) -> OptimizationConfig:
    default = "advisor_packed_bfp8_hifi2_1d" if batch == EMITTED_BATCH else "packed_mlp_bfp8_hifi2_dram_gate40c"
    return OptimizationConfig.named(os.environ.get(PROFILE_ENV, default))


def test_optimized_path_is_independent_and_has_no_host_fallback():
    assert FunctionalDecoder not in OptimizedDecoder.__mro__
    assert OptimizedDecoder.__module__.endswith("optimized_decoder")
    forbidden = ("torch", "from_torch", "to_torch", "tilize", "untilize", "ttnn.reshard")
    methods = (
        OptimizedDecoder._decode_input,
        OptimizedDecoder._decode_norm,
        OptimizedDecoder._qkv_forward,
        OptimizedDecoder._mlp_forward,
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder.decode_forward,
    )
    for method in methods:
        source = inspect.getsource(method)
        for token in forbidden:
            assert token not in source, f"{method.__name__} contains forbidden runtime token {token!r}"

    mlp_source = inspect.getsource(OptimizedDecoder._mlp_forward)
    assert "input_tensor_a_activations=[ttnn.UnaryOpType.SILU]" in mlp_source
    assert "ttnn.silu" not in mlp_source
    assert "self.qkv_weight" in inspect.getsource(OptimizedDecoder._qkv_forward)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_real_weight_decode_qkv_projection_matches_hf(mesh_device):
    config = _config()
    state = _real_state(REPRESENTATIVE_LAYER)
    profile = OptimizationConfig.named("bf16_hifi4_unfused_cache_32c")
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=1,
        optimization_config=profile,
    )
    reference_layer = _hf_layer(state, config, REPRESENTATIVE_LAYER)
    hidden = _recorded_activation(1, seed=32001)
    reference_norm = reference_layer.input_layernorm(hidden[0]).unsqueeze(0)
    normed = decoder._decode_norm(decoder._decode_input(_tt_tensor(hidden, mesh_device)), decoder.input_norm)
    _assert_pcc(reference_norm, _to_host(normed).permute(0, 2, 1, 3), 0.999, "decode sharded RMSNorm")

    q = reference_layer.self_attn.q_proj(reference_norm[0])
    k = reference_layer.self_attn.k_proj(reference_norm[0])
    v = reference_layer.self_attn.v_proj(reference_norm[0])
    reference_qkv = torch.cat((q, k, v), dim=-1).unsqueeze(0)
    actual_qkv = decoder._qkv_forward(normed, mode="decode", seq_len=1)
    if profile.sharded_residual:
        actual_qkv = ttnn.sharded_to_interleaved(actual_qkv, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16)
    actual_qkv = _to_host(actual_qkv).permute(0, 2, 1, 3)
    q_end = config.num_attention_heads * _head_dim(config)
    k_end = q_end + config.num_key_value_heads * _head_dim(config)
    _assert_pcc(reference_qkv[..., :q_end], actual_qkv[..., :q_end], 0.0, "decode DRAM-sharded Q")
    _assert_pcc(reference_qkv[..., q_end:k_end], actual_qkv[..., q_end:k_end], 0.0, "decode DRAM-sharded K")
    _assert_pcc(reference_qkv[..., k_end:], actual_qkv[..., k_end:], 0.0, "decode DRAM-sharded V")
    for start in range(0, reference_qkv.shape[-1], 896):
        _assert_pcc(
            reference_qkv[..., start : start + 896],
            actual_qkv[..., start : start + 896],
            -1.0,
            f"decode QKV DRAM shard start={start}",
        )
    _assert_pcc(reference_qkv, actual_qkv, 0.999, "decode DRAM-sharded QKV")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_default_optimized_synthetic_non_aligned_prefill_decode_and_batch32(mesh_device):
    config = _config()
    profile = _profile_from_env(batch=EMITTED_BATCH)
    state = _synthetic_state(config)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
        optimization_config=profile,
    )
    reference_layer = _hf_layer(state, config, REPRESENTATIVE_LAYER)

    for seq_len in (EMITTED_PREFILL_SEQUENCE, 33):
        hidden = torch.randn(
            (1, EMITTED_BATCH, seq_len, config.hidden_size),
            generator=torch.Generator().manual_seed(5100 + seq_len),
            dtype=torch.bfloat16,
        )
        reference, reference_key, reference_value, _ = _reference_layer(
            reference_layer, hidden, config, REPRESENTATIVE_LAYER
        )
        key_cache, value_cache = _empty_caches(
            config,
            mesh_device,
            batch=EMITTED_BATCH,
            max_cache_len=EMITTED_CACHE_LENGTH,
            dtype=profile.kv_cache_dtype,
        )
        actual = decoder.prefill_forward(_tt_tensor(hidden, mesh_device), key_cache, value_cache)
        _assert_pcc(
            reference,
            _to_host(actual),
            FUNCTIONAL_PREFILL_PCC,
            f"optimized synthetic prefill seq={seq_len}",
        )
        _assert_pcc(
            reference_key,
            _to_host(key_cache)[:, :, :seq_len, :],
            SYNTHETIC_CACHE_PCC,
            "optimized key fill",
        )
        _assert_pcc(
            reference_value,
            _to_host(value_cache)[:, :, :seq_len, :],
            SYNTHETIC_CACHE_PCC,
            "optimized value fill",
        )

    generator = torch.Generator().manual_seed(250320)
    prefill_hidden = torch.randn(
        (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    _, _, _, reference_cache = _reference_layer(reference_layer, prefill_hidden, config, REPRESENTATIVE_LAYER)
    key_cache, value_cache = _empty_caches(
        config,
        mesh_device,
        batch=EMITTED_BATCH,
        max_cache_len=EMITTED_CACHE_LENGTH,
        dtype=profile.kv_cache_dtype,
    )
    decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache)
    decode_hidden = torch.randn((1, EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    reference, reference_key, reference_value, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        REPRESENTATIVE_LAYER,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    actual = decoder.decode_forward(
        _tt_tensor(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    _assert_pcc(
        reference_key,
        _to_host(key_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :],
        SYNTHETIC_CACHE_PCC,
        "optimized decode key append",
    )
    _assert_pcc(
        reference_value,
        _to_host(value_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :],
        SYNTHETIC_CACHE_PCC,
        "optimized decode value append",
    )
    _assert_pcc(reference, _to_host(actual), FUNCTIONAL_DECODE_PCC, "optimized synthetic decode batch32")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_default_optimized_real_layer32_recorded_activation_repeated_decode_and_determinism(mesh_device):
    config = _config()
    profile = _profile_from_env()
    layer_idx = REPRESENTATIVE_LAYER
    state = _real_state(layer_idx)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=1,
        optimization_config=profile,
    )
    reference_layer = _hf_layer(state, config, layer_idx)
    prefill_hidden = _recorded_activation(EMITTED_PREFILL_SEQUENCE)
    reference, reference_key, reference_value, reference_cache = _reference_layer(
        reference_layer, prefill_hidden, config, layer_idx
    )
    key_cache, value_cache = _empty_caches(
        config,
        mesh_device,
        batch=1,
        max_cache_len=EMITTED_CACHE_LENGTH,
        dtype=profile.kv_cache_dtype,
    )
    actual = decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache)
    prefill_pcc = _assert_pcc(reference, _to_host(actual), FUNCTIONAL_PREFILL_PCC, "real-weight layer32 prefill")
    key_pcc = _assert_pcc(
        reference_key,
        _to_host(key_cache)[:, :, :EMITTED_PREFILL_SEQUENCE, :],
        FUNCTIONAL_CACHE_PCC,
        "real key cache",
    )
    value_pcc = _assert_pcc(
        reference_value,
        _to_host(value_cache)[:, :, :EMITTED_PREFILL_SEQUENCE, :],
        FUNCTIONAL_CACHE_PCC,
        "real value cache",
    )

    decode_inputs = _recorded_activation(3, seed=250323)
    decode_outputs = []
    decode_pccs = []
    for offset in range(3):
        hidden = decode_inputs[:, :, offset : offset + 1, :]
        reference, _, _, reference_cache = _reference_layer(
            reference_layer,
            hidden,
            config,
            layer_idx,
            start_pos=EMITTED_PREFILL_SEQUENCE + offset,
            cache=reference_cache,
        )
        actual = decoder.decode_forward(
            _tt_tensor(hidden, mesh_device),
            key_cache,
            value_cache,
            current_pos=EMITTED_PREFILL_SEQUENCE + offset,
        )
        actual_host = _to_host(actual)
        decode_outputs.append(actual_host)
        decode_pccs.append(
            _assert_pcc(
                reference,
                actual_host,
                FUNCTIONAL_DECODE_PCC,
                f"real-activation repeated decode step={offset}",
            )
        )

    key_cache_2, value_cache_2 = _empty_caches(
        config,
        mesh_device,
        batch=1,
        max_cache_len=EMITTED_CACHE_LENGTH,
        dtype=profile.kv_cache_dtype,
    )
    decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache_2, value_cache_2)
    replay_outputs = []
    for offset in range(3):
        actual = decoder.decode_forward(
            _tt_tensor(decode_inputs[:, :, offset : offset + 1, :], mesh_device),
            key_cache_2,
            value_cache_2,
            current_pos=EMITTED_PREFILL_SEQUENCE + offset,
        )
        replay_outputs.append(_to_host(actual))
    for offset, (first, second) in enumerate(zip(decode_outputs, replay_outputs)):
        assert torch.equal(first, second), f"decode replay at step {offset} is not bitwise deterministic"

    print(
        "REAL_PROFILE_RESULT",
        json.dumps(
            {
                "profile": profile.name,
                "layer": layer_idx,
                "activation": "deterministic_recorded_random",
                "prefill_pcc": prefill_pcc,
                "key_cache_pcc": key_pcc,
                "value_cache_pcc": value_pcc,
                "decode_pccs": decode_pccs,
                "deterministic": True,
            },
            sort_keys=True,
        ),
    )


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_default_batch32_advisor_real_layer32_matches_functional_bar(mesh_device):
    """Real-weight gate for the advisor-selected batch-32 fast path."""

    config = _config()
    profile = _profile_from_env(batch=EMITTED_BATCH)
    state = _real_state(REPRESENTATIVE_LAYER)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
        optimization_config=profile,
    )
    reference_layer = _hf_layer(state, config, REPRESENTATIVE_LAYER)

    one_user_prefill = _recorded_activation(EMITTED_PREFILL_SEQUENCE, seed=250326)
    reference, reference_key, reference_value, reference_cache = _reference_layer(
        reference_layer, one_user_prefill, config, REPRESENTATIVE_LAYER
    )
    batched_prefill = one_user_prefill.expand(1, EMITTED_BATCH, -1, -1).contiguous()
    key_cache, value_cache = _empty_caches(
        config,
        mesh_device,
        batch=EMITTED_BATCH,
        max_cache_len=EMITTED_CACHE_LENGTH,
        dtype=profile.kv_cache_dtype,
    )
    actual = decoder.prefill_forward(_tt_tensor(batched_prefill, mesh_device), key_cache, value_cache)
    prefill_pcc = _assert_pcc(
        reference,
        _to_host(actual)[:, :1],
        FUNCTIONAL_PREFILL_PCC,
        "real layer32 batch32 advisor prefill user0",
    )
    _assert_pcc(
        reference_key,
        _to_host(key_cache)[:1, :, :EMITTED_PREFILL_SEQUENCE, :],
        FUNCTIONAL_CACHE_PCC,
        "real layer32 batch32 advisor key cache user0",
    )
    _assert_pcc(
        reference_value,
        _to_host(value_cache)[:1, :, :EMITTED_PREFILL_SEQUENCE, :],
        FUNCTIONAL_CACHE_PCC,
        "real layer32 batch32 advisor value cache user0",
    )

    one_user_decode = _recorded_activation(1, seed=250327)
    reference, _, _, _ = _reference_layer(
        reference_layer,
        one_user_decode,
        config,
        REPRESENTATIVE_LAYER,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    batched_decode = one_user_decode.expand(1, EMITTED_BATCH, -1, -1).contiguous()
    actual = decoder.decode_forward(
        _tt_tensor(batched_decode, mesh_device),
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    decode_pcc = _assert_pcc(
        reference,
        _to_host(actual)[:, :1],
        FUNCTIONAL_DECODE_PCC,
        "real layer32 batch32 advisor decode user0",
    )
    print(
        "BATCH32_ADVISOR_REAL_RESULT",
        json.dumps(
            {"profile": profile.name, "prefill_pcc": prefill_pcc, "decode_pcc": decode_pcc},
            sort_keys=True,
        ),
    )


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_real_layer32_profile_matches_functional_bar(mesh_device):
    if PROFILE_ENV not in os.environ:
        pytest.skip(f"Set {PROFILE_ENV} for an explicit candidate run")
    config = _config()
    profile = _profile_from_env()
    state = _real_state(REPRESENTATIVE_LAYER)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=1,
        optimization_config=profile,
    )
    reference_layer = _hf_layer(state, config, REPRESENTATIVE_LAYER)
    generator = torch.Generator().manual_seed(32)
    prefill_hidden = torch.randn(
        (1, 1, EMITTED_PREFILL_SEQUENCE, config.hidden_size), generator=generator, dtype=torch.bfloat16
    )
    reference, _, _, reference_cache = _reference_layer(reference_layer, prefill_hidden, config, REPRESENTATIVE_LAYER)
    key_cache, value_cache = _empty_caches(
        config,
        mesh_device,
        batch=1,
        max_cache_len=EMITTED_CACHE_LENGTH,
        dtype=profile.kv_cache_dtype,
    )
    actual = decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache)
    prefill_pcc = _assert_pcc(reference, _to_host(actual), FUNCTIONAL_PREFILL_PCC, "real layer32 prefill")
    decode_hidden = torch.randn((1, 1, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    reference, reference_key, reference_value, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        REPRESENTATIVE_LAYER,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    actual = decoder.decode_forward(
        _tt_tensor(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    _assert_pcc(
        reference_key,
        _to_host(key_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :],
        FUNCTIONAL_CACHE_PCC,
        "real layer32 decode key append",
    )
    _assert_pcc(
        reference_value,
        _to_host(value_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :],
        FUNCTIONAL_CACHE_PCC,
        "real layer32 decode value append",
    )
    decode_pcc = _assert_pcc(reference, _to_host(actual), FUNCTIONAL_DECODE_PCC, "real layer32 decode")
    print(
        "LAYER32_PROFILE_RESULT",
        json.dumps(
            {"profile": profile.name, "prefill_pcc": prefill_pcc, "decode_pcc": decode_pcc},
            sort_keys=True,
        ),
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_warmed_prefill_and_traced_decode_perf(mesh_device):
    if os.environ.get(PERF_ENV) != "1":
        pytest.skip(f"Set {PERF_ENV}=1 for the serialized performance run")
    decoder_kind = os.environ.get(PERF_DECODER_ENV, "optimized")
    batch = int(os.environ.get(PERF_BATCH_ENV, "1"))
    repetitions = int(os.environ.get(PERF_REPS_ENV, "50"))
    config = _config()
    state = _real_state(REPRESENTATIVE_LAYER)
    profile = _profile_from_env(batch=batch)
    if decoder_kind == "functional":
        decoder = FunctionalDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            batch=batch,
        )
        cache_dtype = ttnn.bfloat16
    elif decoder_kind == "optimized":
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            batch=batch,
            optimization_config=profile,
        )
        cache_dtype = profile.kv_cache_dtype
    else:
        raise ValueError(f"Unknown {PERF_DECODER_ENV}={decoder_kind!r}")

    try:
        from tracy import signpost
    except ImportError:

        def signpost(*_args, **_kwargs):
            return None

    prefill_hidden = _recorded_activation(EMITTED_PREFILL_SEQUENCE).expand(batch, -1, -1, -1)
    prefill_hidden = prefill_hidden.permute(1, 0, 2, 3).contiguous()
    key_cache, value_cache = _empty_caches(
        config,
        mesh_device,
        batch=batch,
        max_cache_len=EMITTED_CACHE_LENGTH,
        dtype=cache_dtype,
    )
    tt_prefill = _tt_tensor(prefill_hidden, mesh_device)
    decoder.prefill_forward(tt_prefill, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_PREFILL")
    start = time.perf_counter()
    decoder.prefill_forward(tt_prefill, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    prefill_ms = (time.perf_counter() - start) * 1000.0
    signpost("PERF_PREFILL_END")

    decode_hidden = _recorded_activation(1, seed=250324).expand(batch, -1, -1, -1).permute(1, 0, 2, 3).contiguous()
    tt_decode = _tt_tensor(decode_hidden, mesh_device)
    decoder.decode_forward(
        tt_decode,
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    ttnn.synchronize_device(mesh_device)

    # Slice bounds and cache indices participate in program-cache signatures.
    # Warm the exact position used by capture so tracing cannot trigger a
    # compile-time host write.
    trace_pos = EMITTED_PREFILL_SEQUENCE + 1
    decoder.decode_forward(
        tt_decode,
        key_cache,
        value_cache,
        current_pos=trace_pos,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = decoder.decode_forward(
        tt_decode,
        key_cache,
        value_cache,
        current_pos=trace_pos,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for _ in range(5):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)

    signpost("PERF_DECODE")
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    signpost("PERF_DECODE_END")
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        timings.append((time.perf_counter() - start) * 1000.0)
    trace_host = _to_host(trace_output).clone()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    trace_replay_host = _to_host(trace_output).clone()
    assert torch.equal(trace_host, trace_replay_host), "same-input trace replay is not bitwise deterministic"

    refresh_hidden = _recorded_activation(1, seed=250325).expand(batch, -1, -1, -1).permute(1, 0, 2, 3).contiguous()
    tt_refresh_host = ttnn.from_torch(refresh_hidden, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn.copy_host_to_device_tensor(tt_refresh_host, tt_decode, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    refreshed_trace_host = _to_host(trace_output).clone()
    assert not torch.equal(trace_host, refreshed_trace_host), "trace replay ignored an updated device input"
    ttnn.release_trace(mesh_device, trace_id)

    result = {
        "decoder": decoder_kind,
        "profile": profile.name if decoder_kind == "optimized" else "functional_bf16_dram",
        "batch": batch,
        "logical_users": batch,
        "tile_padded_rows": 32 * ((batch + 31) // 32),
        "prefill_seq_len": EMITTED_PREFILL_SEQUENCE,
        "prefill_ms": prefill_ms,
        "decode_ms_per_token_e2e_mean": sum(timings) / len(timings),
        "decode_ms_per_token_e2e_min": min(timings),
        "trace_repetitions": repetitions,
        "trace_output_finite": bool(torch.isfinite(trace_host).all()),
        "trace_same_input_bitwise_deterministic": True,
        "trace_updated_input_observed": True,
        "trace_input_refreshes": 1,
        "trace_readbacks_outside_timing": 3,
        "kv_cache_dtype": str(cache_dtype),
    }
    print("PERF_RESULT", json.dumps(result, sort_keys=True))
    assert result["trace_output_finite"]
