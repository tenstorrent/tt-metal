# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import (
    ADVERTISED_CONTEXT,
    MODEL_ID,
    FunctionalDecoder,
)
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import comp_pcc

REPRESENTATIVE_LAYERS = (0, 1, 4)
REAL_REVISION = "d11e61a842617a22dc328552fa5bb86231ee4f37"


def _config():
    config = AutoConfig.from_pretrained(MODEL_ID, revision=REAL_REVISION)
    assert config.max_position_embeddings == ADVERTISED_CONTEXT
    return config


def _key(layer_idx, suffix):
    return f"model.layers.{layer_idx}.{suffix}"


def _randn(generator, *shape, scale=0.01, mean=0.0):
    return torch.randn(*shape, generator=generator, dtype=torch.bfloat16).mul_(scale).add_(mean)


def _synthetic_state(config, layer_idx, *, sparse_weights=False):
    """Deterministic full target shapes; sparse experts default to zero for CI memory stability."""
    generator = torch.Generator().manual_seed(20260728 + layer_idx)
    use_recorded_sparse_stats = config.mlp_layer_types[layer_idx] == "sparse" and sparse_weights
    attention_scales = (
        {"q": 0.0307369, "k": 0.0504577, "v": 0.0132027, "o": 0.0206136}
        if use_recorded_sparse_stats
        else {"q": 0.01, "k": 0.01, "v": 0.01, "o": 0.01}
    )
    state = {
        _key(layer_idx, "input_layernorm.weight"): (
            _randn(generator, config.hidden_size, scale=0.0468898, mean=0.2410394)
            if use_recorded_sparse_stats
            else torch.ones(config.hidden_size, dtype=torch.bfloat16)
        ),
        _key(layer_idx, "self_attn.q_proj.weight"): _randn(
            generator,
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            scale=attention_scales["q"],
        ),
        _key(layer_idx, "self_attn.k_proj.weight"): _randn(
            generator,
            config.num_key_value_heads * config.head_dim,
            config.hidden_size,
            scale=attention_scales["k"],
        ),
        _key(layer_idx, "self_attn.v_proj.weight"): _randn(
            generator,
            config.num_key_value_heads * config.head_dim,
            config.hidden_size,
            scale=attention_scales["v"],
        ),
        _key(layer_idx, "self_attn.o_proj.weight"): _randn(
            generator,
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            scale=attention_scales["o"],
        ),
    }
    if config.mlp_layer_types[layer_idx] == "dense":
        state.update(
            {
                _key(layer_idx, "mlp.gate_proj.weight"): _randn(
                    generator, config.prefix_dense_intermediate_size, config.hidden_size
                ),
                _key(layer_idx, "mlp.up_proj.weight"): _randn(
                    generator, config.prefix_dense_intermediate_size, config.hidden_size
                ),
                _key(layer_idx, "mlp.down_proj.weight"): _randn(
                    generator, config.hidden_size, config.prefix_dense_intermediate_size
                ),
            }
        )
    else:
        state[_key(layer_idx, "mlp.gate.weight")] = (
            _randn(generator, config.num_experts, config.hidden_size, scale=0.0838776)
            if sparse_weights
            else torch.zeros(config.num_experts, config.hidden_size, dtype=torch.bfloat16)
        )
        shape = (config.num_experts, config.intermediate_size, config.hidden_size)
        down_shape = (config.num_experts, config.hidden_size, config.intermediate_size)
        state[_key(layer_idx, "mlp.experts.gate_up_proj")] = torch.cat(
            (
                (
                    _randn(generator, *shape, scale=0.0218788)
                    if sparse_weights
                    else torch.zeros(shape, dtype=torch.bfloat16)
                ),
                (
                    _randn(generator, *shape, scale=0.0214365)
                    if sparse_weights
                    else torch.zeros(shape, dtype=torch.bfloat16)
                ),
            ),
            dim=1,
        )
        state[_key(layer_idx, "mlp.experts.down_proj")] = (
            _randn(generator, *down_shape, scale=0.0268128)
            if sparse_weights
            else torch.zeros(down_shape, dtype=torch.bfloat16)
        )
    return state


def _real_layer_one_state():
    explicit = os.environ.get("NORTH_MINI_REAL_WEIGHT_DIR")
    roots = [Path(explicit)] if explicit else []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.extend((Path(hf_home), Path(hf_home) / "hub"))
    roots.append(Path("/huggingface/hub"))
    snapshot = next(
        (
            path
            for root in roots
            for path in root.glob(f"models--CohereLabs--North-Mini-Code-1.0/snapshots/{REAL_REVISION}")
            if path.is_dir()
        ),
        None,
    )
    if snapshot is None:
        pytest.skip("North-Mini checkpoint not cached; set NORTH_MINI_REAL_WEIGHT_DIR")
    shards = [snapshot / f"model-{index:05d}-of-00049.safetensors" for index in (1, 2)]
    if not all(shard.is_file() for shard in shards):
        pytest.skip("North-Mini layer-1 shards 1 and 2 are not cached")
    prefix = "model.layers.1."
    state = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            state.update({key: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)})
    return state


def _to_tt(tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
    )


def _to_host_tt(tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
    )


def _page_table(batch, blocks_per_user):
    """Non-identity physical placement, deterministic for cache-address checks."""
    blocks = batch * blocks_per_user
    return torch.arange(blocks - 1, -1, -1, dtype=torch.int32).reshape(batch, blocks_per_user)


def _rope_interleaved(tensor, positions, config):
    inv = 1.0 / (
        float(config.rope_parameters["rope_theta"])
        ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
    )
    frequencies = positions.float().unsqueeze(-1) * inv
    cos = torch.repeat_interleave(frequencies, 2, dim=-1).cos().to(tensor.dtype)
    sin = torch.repeat_interleave(frequencies, 2, dim=-1).sin().to(tensor.dtype)
    rotated = torch.stack((-tensor[..., 1::2], tensor[..., ::2]), dim=-1).flatten(-2)
    return tensor * cos[:, None, :, :] + rotated * sin[:, None, :, :]


def _dense_reference(hidden, positions, state, config, *, cache=None):
    prefix = "model.layers.0."
    normalized = (hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)).to(
        torch.bfloat16
    )
    normalized *= state[prefix + "input_layernorm.weight"]
    batch, sequence, _ = hidden.shape
    query = F.linear(normalized, state[prefix + "self_attn.q_proj.weight"])
    key = F.linear(normalized, state[prefix + "self_attn.k_proj.weight"])
    value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    query = query.view(batch, sequence, config.num_attention_heads, config.head_dim).transpose(1, 2)
    key = key.view(batch, sequence, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    value = value.view(batch, sequence, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    query = _rope_interleaved(query, positions, config)
    key = _rope_interleaved(key, positions, config)
    if cache is not None:
        key = torch.cat((cache[0], key), dim=2)
        value = torch.cat((cache[1], value), dim=2)
    repeated_key = key.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=1)
    repeated_value = value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=1)
    scores = torch.matmul(query.float(), repeated_key.float().transpose(-2, -1)) / math.sqrt(config.head_dim)
    past = key.shape[2] - sequence
    allowed = torch.arange(key.shape[2])[None, :] <= torch.arange(sequence)[:, None] + past
    scores.masked_fill_(~allowed, -torch.inf)
    probs = torch.softmax(scores, dim=-1).to(torch.bfloat16)
    attention = torch.matmul(probs, repeated_value).transpose(1, 2).reshape(batch, sequence, -1)
    attention = F.linear(attention, state[prefix + "self_attn.o_proj.weight"])
    gate = F.linear(normalized, state[prefix + "mlp.gate_proj.weight"])
    up = F.linear(normalized, state[prefix + "mlp.up_proj.weight"])
    mlp = F.linear(
        F.silu(gate) * up,
        state[prefix + "mlp.down_proj.weight"],
    )
    return hidden + attention + mlp, (key, value)


def _attention_reference(hidden, positions, state, config, layer_idx, *, cache=None):
    """Reference the common attention/residual path for every decoder kind."""
    prefix = f"model.layers.{layer_idx}."
    normalized = (hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)).to(
        torch.bfloat16
    )
    normalized *= state[prefix + "input_layernorm.weight"]
    batch, sequence, _ = hidden.shape
    query = F.linear(normalized, state[prefix + "self_attn.q_proj.weight"])
    key = F.linear(normalized, state[prefix + "self_attn.k_proj.weight"])
    value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    query = query.view(batch, sequence, config.num_attention_heads, config.head_dim).transpose(1, 2)
    key = key.view(batch, sequence, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    value = value.view(batch, sequence, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    use_rope = config.layer_types[layer_idx] == "sliding_attention" or (
        config.mlp_layer_types[layer_idx] == "dense" and config.prefix_dense_sliding_window_pattern == 1
    )
    if use_rope:
        query = _rope_interleaved(query, positions, config)
        key = _rope_interleaved(key, positions, config)
    if cache is not None:
        key = torch.cat((cache[0], key), dim=2)
        value = torch.cat((cache[1], value), dim=2)
    repeated_key = key.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=1)
    repeated_value = value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=1)
    scores = torch.matmul(query.float(), repeated_key.float().transpose(-2, -1)) / math.sqrt(config.head_dim)
    past = key.shape[2] - sequence
    query_index = torch.arange(sequence)[:, None] + past
    key_index = torch.arange(key.shape[2])[None, :]
    allowed = key_index <= query_index
    if config.layer_types[layer_idx] == "sliding_attention":
        allowed &= key_index > query_index - config.sliding_window
    scores.masked_fill_(~allowed, -torch.inf)
    probs = torch.softmax(scores, dim=-1).to(torch.bfloat16)
    attention = torch.matmul(probs, repeated_value).transpose(1, 2).reshape(batch, sequence, -1)
    attention = F.linear(attention, state[prefix + "self_attn.o_proj.weight"])
    return hidden + attention, (key, value)


def _normalized(hidden, state, config, layer_idx):
    prefix = f"model.layers.{layer_idx}."
    result = (hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)).to(
        torch.bfloat16
    )
    return result * state[prefix + "input_layernorm.weight"]


def _sparse_moe_reference(normalized, state, config, layer_idx, *, flat_indices=None):
    """Reference selected tokens without evaluating inactive experts."""
    prefix = f"model.layers.{layer_idx}."
    flat = normalized.reshape(-1, config.hidden_size)
    if flat_indices is not None:
        flat = flat[torch.as_tensor(flat_indices)]
    logits = F.linear(flat, state[prefix + "mlp.gate.weight"])
    scores, experts = torch.topk(logits, config.num_experts_per_tok, dim=-1)
    scores = torch.sigmoid(scores)
    fused = state[prefix + "mlp.experts.gate_up_proj"]
    down = state[prefix + "mlp.experts.down_proj"]
    result = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for route in range(config.num_experts_per_tok):
            expert = int(experts[token, route])
            gate = F.linear(flat[token], fused[expert, : config.intermediate_size])
            up = F.linear(flat[token], fused[expert, config.intermediate_size :])
            contribution = F.linear(F.silu(gate) * up, down[expert])
            result[token] += contribution * scores[token, route]
    return result, experts


def _project_split_qkv(hidden, positions, state, config, layer_idx):
    prefix = f"model.layers.{layer_idx}."
    normalized = _normalized(hidden, state, config, layer_idx)
    batch, sequence, _ = hidden.shape
    query = F.linear(normalized, state[prefix + "self_attn.q_proj.weight"])
    key = F.linear(normalized, state[prefix + "self_attn.k_proj.weight"])
    value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    query = query.view(batch, sequence, config.num_attention_heads, config.head_dim).transpose(1, 2)
    key = key.view(batch, sequence, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    value = value.view(batch, sequence, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    query = _rope_interleaved(query, positions, config)
    key = _rope_interleaved(key, positions, config)
    query = torch.cat((query[..., ::2], query[..., 1::2]), dim=-1)
    key = torch.cat((key[..., ::2], key[..., 1::2]), dim=-1)
    return normalized, query, key, value


def _assert_pcc(name, reference, actual, threshold=0.995):
    passed, message = comp_pcc(reference.float(), actual.float(), pcc=threshold)
    print(f"{name}: {message}")
    assert passed, f"{name}: {message}"


def _decode_inputs(decoder, config, mesh_device, positions):
    cos, sin = decoder.build_rope_rows(positions, hf_config=config, decode=True)
    return (
        _to_tt(torch.tensor(positions, dtype=torch.int32), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        _to_tt(cos, mesh_device),
        _to_tt(sin, mesh_device),
    )


def test_contract_and_runtime_fallback_audit():
    assert issubclass(FunctionalDecoder, LightweightModule)
    assert ADVERTISED_CONTEXT == _config().max_position_embeddings
    prefill = inspect.signature(FunctionalDecoder.prefill_forward)
    decode = inspect.signature(FunctionalDecoder.decode_forward)
    assert {"key_cache", "value_cache", "page_table", "position_cos", "position_sin"} <= set(prefill.parameters)
    assert {"page_table", "current_positions", "position_cos", "position_sin"} <= set(decode.parameters)
    runtime_source = "\n".join(
        inspect.getsource(method)
        for method in (
            FunctionalDecoder.prefill_forward,
            FunctionalDecoder.decode_forward,
            FunctionalDecoder._attention_prefill,
            FunctionalDecoder._attention_decode,
            FunctionalDecoder._dense_mlp,
            FunctionalDecoder._sparse_moe,
            FunctionalDecoder._sparse_moe_chunk,
        )
    )
    for forbidden in ("import torch", "from_torch", "to_torch", "program_config"):
        assert forbidden not in runtime_source


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [1, 31, 32, 33, 65])
def test_dense_paged_prefill_non_aligned_matches_reference(mesh_device, seq_len):
    config = _config()
    state = _synthetic_state(config, 0)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=0, mesh_device=mesh_device, batch=1, max_cache_len=96
    )
    generator = torch.Generator().manual_seed(8000 + seq_len)
    hidden = (_randn(generator, 1, seq_len, config.hidden_size, scale=0.02)).unsqueeze(0)
    reference, _ = _dense_reference(hidden.squeeze(0), torch.arange(seq_len).reshape(1, -1), state, config)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    table = _page_table(1, 3)
    page_table = _to_tt(table, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = decoder.build_rope_rows(torch.arange(seq_len), hf_config=config)
    actual = decoder.prefill_forward(
        _to_tt(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    _assert_pcc(f"dense-prefill-{seq_len}", reference, ttnn.to_torch(actual).squeeze(0))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dense_paged_decode_trace_replay_matches_reference(mesh_device):
    config = _config()
    state = _synthetic_state(config, 0)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=0, mesh_device=mesh_device, batch=1, max_cache_len=96
    )
    generator = torch.Generator().manual_seed(9001)
    prefill_len = 33
    prefill_hidden = _randn(generator, 1, prefill_len, config.hidden_size, scale=0.02)
    _, reference_cache = _dense_reference(prefill_hidden, torch.arange(prefill_len).reshape(1, -1), state, config)
    decode_hidden = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    reference, _ = _dense_reference(decode_hidden, torch.tensor([[prefill_len]]), state, config, cache=reference_cache)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 3), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = decoder.build_rope_rows(torch.arange(prefill_len), hf_config=config)
    decoder.prefill_forward(
        _to_tt(prefill_hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    hidden_tt = _to_tt(decode_hidden.unsqueeze(0), mesh_device)
    current, cos_tt, sin_tt = _decode_inputs(decoder, config, mesh_device, [prefill_len])
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos_tt,
        position_sin=sin_tt,
    )
    decoder.decode_forward(hidden_tt, **kwargs)  # compile
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    actual = decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc("dense-traced-decode", reference, ttnn.to_torch(actual).squeeze(0))
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_batch_two_prefill_and_permuted_physical_cache(mesh_device):
    config = _config()
    state = _synthetic_state(config, 0)
    batch, sequence = 2, 33
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=0, mesh_device=mesh_device, batch=batch, max_cache_len=64
    )
    generator = torch.Generator().manual_seed(9233)
    hidden = _randn(generator, batch, sequence, config.hidden_size, scale=0.02)
    reference, reference_cache = _dense_reference(hidden, torch.arange(sequence).reshape(1, -1), state, config)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    table = _page_table(batch, 2)
    page_table = _to_tt(table, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = decoder.build_rope_rows(torch.arange(sequence), hf_config=config)
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    _assert_pcc("batch-two-prefill", reference, ttnn.to_torch(actual).squeeze(0))

    physical_key = ttnn.to_torch(key_cache)
    physical_value = ttnn.to_torch(value_cache)
    expected_key = torch.cat((reference_cache[0][..., ::2], reference_cache[0][..., 1::2]), dim=-1)
    for user in range(batch):
        for logical_block in range(2):
            start = logical_block * decoder.page_size
            stop = min(start + decoder.page_size, sequence)
            physical_block = int(table[user, logical_block])
            _assert_pcc(
                f"key-cache-user{user}-block{logical_block}",
                expected_key[user, :, start:stop],
                physical_key[physical_block, :, : stop - start],
            )
            _assert_pcc(
                f"value-cache-user{user}-block{logical_block}",
                reference_cache[1][user, :, start:stop],
                physical_value[physical_block, :, : stop - start],
            )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_random_nonzero_decode_positions_update_expected_physical_slots_and_are_deterministic(mesh_device):
    config = _config()
    state = _synthetic_state(config, 0)
    batch, positions = 4, [5, 17, 31, 63]
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=0, mesh_device=mesh_device, batch=batch, max_cache_len=64
    )
    generator = torch.Generator().manual_seed(9463)
    hidden = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    table = _page_table(batch, 2)
    page_table = _to_tt(table, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, positions)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos,
        position_sin=sin,
    )
    first = ttnn.to_torch(decoder.decode_forward(_to_tt(hidden.unsqueeze(0), mesh_device), **kwargs))
    second = ttnn.to_torch(decoder.decode_forward(_to_tt(hidden.unsqueeze(0), mesh_device), **kwargs))
    assert torch.equal(first, second)

    prefix = "model.layers.0."
    normalized = (hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)).to(
        torch.bfloat16
    )
    normalized *= state[prefix + "input_layernorm.weight"]
    expected_key = F.linear(normalized, state[prefix + "self_attn.k_proj.weight"])
    expected_key = expected_key.view(batch, 1, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    expected_key = _rope_interleaved(expected_key, torch.tensor(positions).reshape(batch, 1), config)
    expected_key = torch.cat((expected_key[..., ::2], expected_key[..., 1::2]), dim=-1)
    expected_value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    expected_value = expected_value.view(batch, 1, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    physical_key = ttnn.to_torch(key_cache)
    physical_value = ttnn.to_torch(value_cache)
    for user, position in enumerate(positions):
        physical_block = int(table[user, position // decoder.page_size])
        slot = position % decoder.page_size
        _assert_pcc(f"decode-key-slot-{user}", expected_key[user, :, 0], physical_key[physical_block, :, slot])
        _assert_pcc(f"decode-value-slot-{user}", expected_value[user, :, 0], physical_value[physical_block, :, slot])


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_serving_batch_32_paged_decode_trace_replay_matches_reference(mesh_device):
    config = _config()
    state = _synthetic_state(config, 0)
    batch = 32
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=0, mesh_device=mesh_device, batch=batch, max_cache_len=32
    )
    generator = torch.Generator().manual_seed(9032)
    hidden = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    reference, _ = _dense_reference(
        hidden,
        torch.zeros(batch, 1, dtype=torch.long),
        state,
        config,
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(batch, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0] * batch)
    hidden_tt = _to_tt(hidden.unsqueeze(0), mesh_device)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos,
        position_sin=sin,
    )
    decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    actual = decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc("serving-batch-32-traced-decode", reference, ttnn.to_torch(actual).squeeze(0))
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", REPRESENTATIVE_LAYERS)
def test_every_meaningful_layer_kind_executes(mesh_device, layer_idx):
    config = _config()
    state = _synthetic_state(config, layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=layer_idx, mesh_device=mesh_device, batch=1, max_cache_len=32
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    hidden = _to_tt(torch.zeros(1, 1, 1, config.hidden_size, dtype=torch.bfloat16), mesh_device)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0])
    output = decoder.decode_forward(
        hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos if decoder.use_rope else None,
        position_sin=sin if decoder.use_rope else None,
    )
    assert tuple(output.shape) == (1, 1, 1, config.hidden_size)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [1, 4])
def test_sparse_layer_kind_paged_prefill_matches_attention_reference(mesh_device, layer_idx):
    config = _config()
    state = _synthetic_state(config, layer_idx)
    sequence = 33
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=layer_idx, mesh_device=mesh_device, batch=1, max_cache_len=64
    )
    generator = torch.Generator().manual_seed(9700 + layer_idx)
    hidden = _randn(generator, 1, sequence, config.hidden_size, scale=0.02)
    reference, _ = _attention_reference(hidden, torch.arange(sequence).reshape(1, -1), state, config, layer_idx)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 2), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = decoder.build_rope_rows(torch.arange(sequence), hf_config=config)
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device) if decoder.use_rope else None,
        position_sin=_to_tt(sin, mesh_device) if decoder.use_rope else None,
    )
    _assert_pcc(f"sparse-layer-{layer_idx}-prefill", reference, ttnn.to_torch(actual).squeeze(0))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_sliding_window_boundary_4097_matches_controlled_reference(mesh_device):
    """Zero Q/K makes exact 4096-wide causal-window semantics cheap to reference."""
    config = _config()
    layer_idx, sequence = 1, config.sliding_window + 1
    state = _synthetic_state(config, layer_idx)
    prefix = f"model.layers.{layer_idx}."
    state[prefix + "self_attn.q_proj.weight"].zero_()
    state[prefix + "self_attn.k_proj.weight"].zero_()
    generator = torch.Generator().manual_seed(14097)
    hidden = _randn(generator, 1, sequence, config.hidden_size, scale=0.02)
    normalized = (hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)).to(
        torch.bfloat16
    )
    normalized *= state[prefix + "input_layernorm.weight"]
    value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    value = value.view(1, sequence, config.num_key_value_heads, config.head_dim)
    prefix_sum = F.pad(value.float().cumsum(dim=1), (0, 0, 0, 0, 1, 0))
    end = torch.arange(1, sequence + 1)
    start = torch.clamp(end - config.sliding_window, min=0)
    window_sum = prefix_sum[:, end] - prefix_sum[:, start]
    window_size = (end - start).reshape(1, sequence, 1, 1)
    attended = (window_sum / window_size).to(torch.bfloat16)
    attended = attended.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=2).reshape(
        1, sequence, -1
    )
    reference = hidden + F.linear(attended, state[prefix + "self_attn.o_proj.weight"])

    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=sequence,
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    blocks = math.ceil(sequence / decoder.page_size)
    page_table = _to_tt(_page_table(1, blocks), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = decoder.build_rope_rows(torch.arange(sequence), hf_config=config)
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    _assert_pcc("sliding-window-4097", reference, ttnn.to_torch(actual).squeeze(0))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,sequence,selected_tokens",
    [
        (1, 1025, [0, 1023, 1024]),
        (4, 33, [0, 16, 32]),
    ],
)
def test_nonzero_sparse_prefill_matches_active_expert_reference(mesh_device, layer_idx, sequence, selected_tokens):
    """Exercise routing, score weighting, expert reduction, and chunk ordering."""
    config = _config()
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    prefix = f"model.layers.{layer_idx}."
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        state[prefix + f"self_attn.{projection}.weight"].zero_()
    generator = torch.Generator().manual_seed(16000 + layer_idx + sequence)
    hidden = _randn(generator, 1, sequence, config.hidden_size, scale=0.02)
    normalized = _normalized(hidden, state, config, layer_idx)
    reference_moe, experts = _sparse_moe_reference(
        normalized,
        state,
        config,
        layer_idx,
        flat_indices=selected_tokens,
    )
    assert torch.unique(experts).numel() > config.num_experts_per_tok
    reference = hidden[:, selected_tokens] + reference_moe.reshape(1, len(selected_tokens), -1)

    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=sequence,
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    blocks = math.ceil(sequence / decoder.page_size)
    page_table = _to_tt(_page_table(1, blocks), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = decoder.build_rope_rows(torch.arange(sequence), hf_config=config)
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device) if decoder.use_rope else None,
        position_sin=_to_tt(sin, mesh_device) if decoder.use_rope else None,
    )
    actual = ttnn.to_torch(actual).squeeze(0)[:, selected_tokens]
    _assert_pcc(f"nonzero-sparse-layer-{layer_idx}-prefill", reference, actual)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_sliding_moe_populated_history_dynamic_trace_replay_matches_reference(mesh_device):
    """Update stable trace inputs and decode beyond the 4096-token window."""
    config = _config()
    layer_idx, history = 1, config.sliding_window
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=history + 2,
    )
    generator = torch.Generator().manual_seed(174096)
    past_key = _randn(generator, 1, config.num_key_value_heads, history, config.head_dim, scale=0.01)
    past_value = _randn(generator, 1, config.num_key_value_heads, history, config.head_dim, scale=0.01)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    blocks = math.ceil((history + 2) / decoder.page_size)
    table = _page_table(1, blocks)
    page_table = _to_tt(table, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn.experimental.paged_fill_cache(key_cache, _to_tt(past_key, mesh_device), page_table, batch_idx=0)
    ttnn.experimental.paged_fill_cache(value_cache, _to_tt(past_value, mesh_device), page_table, batch_idx=0)

    hidden_a = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    hidden_b = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    hidden_tt = _to_tt(hidden_a.unsqueeze(0), mesh_device)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [history])
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos,
        position_sin=sin,
    )
    decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    actual = decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.copy_host_to_device_tensor(_to_host_tt(hidden_b.unsqueeze(0), mesh_device), hidden_tt)
        ttnn.copy_host_to_device_tensor(
            _to_host_tt(
                torch.tensor([history + 1], dtype=torch.int32),
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            current,
        )
        next_cos_host, next_sin_host = decoder.build_rope_rows([history + 1], hf_config=config, decode=True)
        ttnn.copy_host_to_device_tensor(_to_host_tt(next_cos_host, mesh_device), cos)
        ttnn.copy_host_to_device_tensor(_to_host_tt(next_sin_host, mesh_device), sin)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

        _, query_b, key_b, value_b = _project_split_qkv(
            hidden_b, torch.tensor([[history + 1]]), state, config, layer_idx
        )
        normalized_a, _, key_a, value_a = _project_split_qkv(
            hidden_a, torch.tensor([[history]]), state, config, layer_idx
        )
        del normalized_a
        all_key = torch.cat((past_key, key_a, key_b), dim=2)[:, :, -config.sliding_window :]
        all_value = torch.cat((past_value, value_a, value_b), dim=2)[:, :, -config.sliding_window :]
        repeated_key = all_key.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=1)
        repeated_value = all_value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=1)
        scores = torch.matmul(query_b.float(), repeated_key.float().transpose(-2, -1))
        scores /= math.sqrt(config.head_dim)
        probs = torch.softmax(scores, dim=-1).to(torch.bfloat16)
        attention = torch.matmul(probs, repeated_value).transpose(1, 2).reshape(1, 1, -1)
        attention = F.linear(attention, state["model.layers.1.self_attn.o_proj.weight"])
        normalized_b = _normalized(hidden_b, state, config, layer_idx)
        moe, _ = _sparse_moe_reference(normalized_b, state, config, layer_idx)
        reference = hidden_b + attention + moe.reshape_as(hidden_b)
        _assert_pcc("sliding-moe-dynamic-trace", reference, ttnn.to_torch(actual).squeeze(0))
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx,batch", [(1, 32), (4, 1)])
def test_nonzero_sparse_dynamic_trace_replay_matches_reference(mesh_device, layer_idx, batch):
    """Replay updated hidden/position/RoPE buffers for served and full-MoE paths."""
    config = _config()
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=32,
    )
    generator = torch.Generator().manual_seed(18000 + layer_idx + batch)
    hidden_a = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    hidden_b = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(batch, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0] * batch)
    hidden_tt = _to_tt(hidden_a.unsqueeze(0), mesh_device)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos if decoder.use_rope else None,
        position_sin=sin if decoder.use_rope else None,
    )
    decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    actual = decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.copy_host_to_device_tensor(_to_host_tt(hidden_b.unsqueeze(0), mesh_device), hidden_tt)
        ttnn.copy_host_to_device_tensor(
            _to_host_tt(
                torch.zeros(batch, dtype=torch.int32),
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            current,
        )
        if decoder.use_rope:
            cos_host, sin_host = decoder.build_rope_rows([0] * batch, hf_config=config, decode=True)
            ttnn.copy_host_to_device_tensor(_to_host_tt(cos_host, mesh_device), cos)
            ttnn.copy_host_to_device_tensor(_to_host_tt(sin_host, mesh_device), sin)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

        prefix = f"model.layers.{layer_idx}."
        normalized = _normalized(hidden_b, state, config, layer_idx)
        value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
        value = value.view(batch, 1, config.num_key_value_heads, config.head_dim)
        attention = value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=2).reshape(
            batch, 1, -1
        )
        attention = F.linear(attention, state[prefix + "self_attn.o_proj.weight"])
        moe, _ = _sparse_moe_reference(normalized, state, config, layer_idx)
        reference = hidden_b + attention + moe.reshape_as(hidden_b)
        _assert_pcc(
            f"nonzero-sparse-layer-{layer_idx}-batch-{batch}-dynamic-trace",
            reference,
            ttnn.to_torch(actual).squeeze(0),
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_real_weight_sliding_moe_decode_matches_reference(mesh_device):
    """Required real-weight gate: official layer 1, paged position zero."""
    config = _config()
    state = _real_layer_one_state()
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=1, mesh_device=mesh_device, batch=1, max_cache_len=32
    )
    generator = torch.Generator().manual_seed(123)
    hidden = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    prefix = "model.layers.1."
    normalized = (hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)).to(
        torch.bfloat16
    )
    normalized *= state[prefix + "input_layernorm.weight"]
    value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    value = value.view(1, 1, config.num_key_value_heads, config.head_dim)
    attention = value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=2).reshape(
        1, 1, -1
    )
    attention = F.linear(attention, state[prefix + "self_attn.o_proj.weight"])
    logits = F.linear(normalized.reshape(1, -1), state[prefix + "mlp.gate.weight"])
    scores, experts = torch.topk(logits, config.num_experts_per_tok, dim=-1)
    print(f"real-layer1-selected-experts: {experts[0].tolist()}")
    scores = torch.sigmoid(scores)
    moe = torch.zeros_like(normalized.reshape(1, -1))
    for topk_index, expert in enumerate(experts[0].tolist()):
        gate = F.linear(normalized.reshape(1, -1), state[f"{prefix}mlp.experts.{expert}.gate_proj.weight"])
        up = F.linear(normalized.reshape(1, -1), state[f"{prefix}mlp.experts.{expert}.up_proj.weight"])
        contribution = F.linear(
            F.silu(gate) * up,
            state[f"{prefix}mlp.experts.{expert}.down_proj.weight"],
        )
        moe += contribution * scores[0, topk_index]
    reference = hidden + attention + moe.reshape_as(hidden)

    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0])
    actual = decoder.decode_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos,
        position_sin=sin,
    )
    _assert_pcc("real-layer1-decode", reference, ttnn.to_torch(actual).squeeze(0))
