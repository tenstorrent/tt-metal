# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import inspect
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (
    REAL_REVISION,
    _assert_pcc,
    _attention_reference,
    _config,
    _decode_inputs,
    _dense_reference,
    _normalized,
    _page_table,
    _randn,
    _rope_interleaved,
    _sparse_moe_reference,
    _synthetic_state,
    _to_host_tt,
    _to_tt,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import FunctionalDecoder
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import OptimizationConfig, OptimizedDecoder


def _position_zero_reference(hidden, state, config, layer_idx):
    prefix = f"model.layers.{layer_idx}."
    normalized = _normalized(hidden, state, config, layer_idx)
    value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    value = value.view(hidden.shape[0], 1, config.num_key_value_heads, config.head_dim)
    attention = value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=2)
    attention = attention.reshape(hidden.shape[0], 1, -1)
    attention = F.linear(attention, state[prefix + "self_attn.o_proj.weight"])
    if config.mlp_layer_types[layer_idx] == "dense":
        return _dense_reference(hidden, torch.zeros(hidden.shape[0], 1, dtype=torch.long), state, config)[0]
    moe, _ = _sparse_moe_reference(normalized, state, config, layer_idx)
    return hidden + attention + moe.reshape_as(hidden)


def _decode_once(decoder, hidden, config, mesh_device):
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(
        _page_table(decoder.batch, math.ceil(decoder.max_cache_len / decoder.page_size)),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0] * decoder.batch)
    return decoder.decode_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos if decoder.use_rope else None,
        position_sin=sin if decoder.use_rope else None,
    )


def _expert_sweep_policy():
    name = os.environ.get("NORTH_MINI_EXPERT_POLICY", "selected")
    if name == "selected":
        return OptimizationConfig()
    if name == "bfp8_lofi":
        return OptimizationConfig(
            expert_gate_up_dtype=ttnn.bfloat8_b,
            expert_down_dtype=ttnn.bfloat8_b,
            dense_expert_gate_up_dtype=ttnn.bfloat8_b,
            dense_expert_down_dtype=ttnn.bfloat8_b,
        )
    if name == "bfp4_lofi":
        return OptimizationConfig(
            expert_gate_up_dtype=ttnn.bfloat4_b,
            expert_down_dtype=ttnn.bfloat4_b,
            dense_expert_gate_up_dtype=ttnn.bfloat4_b,
            dense_expert_down_dtype=ttnn.bfloat4_b,
            expert_gate_up_fidelity=ttnn.MathFidelity.LoFi,
            expert_down_fidelity=ttnn.MathFidelity.LoFi,
        )
    if name == "bfp4_hifi2":
        return OptimizationConfig(
            expert_gate_up_dtype=ttnn.bfloat4_b,
            expert_down_dtype=ttnn.bfloat4_b,
            expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
            expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        )
    if name == "bfp4_gate":
        return OptimizationConfig(
            expert_gate_up_dtype=ttnn.bfloat4_b,
            dense_expert_gate_up_dtype=ttnn.bfloat4_b,
        )
    if name == "bfp4_down":
        return OptimizationConfig(
            expert_down_dtype=ttnn.bfloat4_b,
            dense_expert_down_dtype=ttnn.bfloat4_b,
        )
    if name == "bfp8_hifi2":
        return OptimizationConfig(
            expert_gate_up_dtype=ttnn.bfloat8_b,
            expert_down_dtype=ttnn.bfloat8_b,
            expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
            expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        )
    if name == "bf16_hifi2":
        return OptimizationConfig(
            expert_gate_up_dtype=ttnn.bfloat16,
            expert_down_dtype=ttnn.bfloat16,
            expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
            expert_down_fidelity=ttnn.MathFidelity.HiFi2,
        )
    if name == "bf16_hifi2_auto_grid":
        return OptimizationConfig(
            expert_gate_up_dtype=ttnn.bfloat16,
            expert_down_dtype=ttnn.bfloat16,
            expert_gate_up_fidelity=ttnn.MathFidelity.HiFi2,
            expert_down_fidelity=ttnn.MathFidelity.HiFi2,
            dense_expert_cores=0,
        )
    raise ValueError(f"unknown NORTH_MINI_EXPERT_POLICY={name!r}")


def _real_layer_zero_state():
    return _real_layer_state(0)


def _real_snapshot():
    explicit = os.environ.get("NORTH_MINI_REAL_WEIGHT_DIR")
    roots = [Path(explicit)] if explicit else []
    roots.extend((Path("/huggingface"), Path("/huggingface/hub")))
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
        pytest.skip("North-Mini checkpoint is not cached")
    return snapshot


def _real_layer_state(layer_idx):
    snapshot = _real_snapshot()
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    prefix = f"model.layers.{layer_idx}."
    shard_names = sorted({value for key, value in index["weight_map"].items() if key.startswith(prefix)})
    shards = [snapshot / name for name in shard_names]
    if not shards or not all(shard.is_file() for shard in shards):
        pytest.skip(f"North-Mini layer-{layer_idx} shards are not cached")
    state = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            state.update({key: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)})
    return state


def _real_embedding_hidden(token_id=123, sequence=1):
    shard = _real_snapshot() / "model-00001-of-00049.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_slice("model.embed_tokens.weight")[token_id : token_id + sequence].reshape(1, sequence, -1)


def _real_moe_reference(hidden, state, config, layer_idx, *, positions=None, cache=None, return_cache=False):
    prefix = f"model.layers.{layer_idx}."
    if positions is None:
        positions = torch.arange(hidden.shape[1]).reshape(1, -1).expand(hidden.shape[0], -1)
    attention_residual, updated_cache = _attention_reference(
        hidden,
        positions,
        state,
        config,
        layer_idx,
        cache=cache,
    )
    normalized = _normalized(hidden, state, config, layer_idx)
    flat = normalized.reshape(-1, config.hidden_size)
    logits = F.linear(flat, state[prefix + "mlp.gate.weight"])
    scores, experts = torch.topk(logits, config.num_experts_per_tok, dim=-1)
    scores = torch.sigmoid(scores)
    moe = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for topk_index, expert in enumerate(experts[token].tolist()):
            gate = F.linear(
                flat[token],
                state[f"{prefix}mlp.experts.{expert}.gate_proj.weight"],
            )
            up = F.linear(
                flat[token],
                state[f"{prefix}mlp.experts.{expert}.up_proj.weight"],
            )
            contribution = F.linear(
                F.silu(gate) * up,
                state[f"{prefix}mlp.experts.{expert}.down_proj.weight"],
            )
            moe[token] += contribution.reshape_as(moe[token]) * scores[token, topk_index]
    output = attention_residual + moe.reshape_as(hidden)
    return (output, updated_cache) if return_cache else output


def _real_hidden_at_layer(layer_idx, config, *, sequence=1, token_id=123):
    hidden = _real_embedding_hidden(token_id=token_id, sequence=sequence)
    positions = torch.arange(sequence).reshape(1, -1)
    for previous_layer in range(layer_idx):
        state = _real_layer_state(previous_layer)
        if config.mlp_layer_types[previous_layer] == "dense":
            hidden = _dense_reference(hidden, positions, state, config)[0]
        else:
            hidden = _real_moe_reference(hidden, state, config, previous_layer)
        del state
        gc.collect()
    return hidden


def test_optimized_path_audit():
    assert issubclass(OptimizedDecoder, FunctionalDecoder)
    policy = OptimizationConfig()
    assert policy.expert_gate_up_dtype == policy.expert_down_dtype == ttnn.bfloat8_b
    assert policy.dense_expert_gate_up_dtype == policy.dense_expert_down_dtype == ttnn.bfloat4_b
    assert policy.expert_gate_up_fidelity == policy.expert_down_fidelity == ttnn.MathFidelity.LoFi
    for method_name in (
        "from_state_dict",
        "prefill_forward",
        "decode_forward",
        "_attention_prefill",
        "_attention_decode",
        "_dense_mlp_decode",
        "_dense_mlp_prefill",
        "_sparse_moe_chunk",
        "_sparse_moe_prefill_chunk",
        "_dense_expert_moe_chunk",
    ):
        assert method_name in OptimizedDecoder.__dict__, f"{method_name} would fall back to functional code"
    runtime_source = "\n".join(
        inspect.getsource(OptimizedDecoder.__dict__[name])
        for name in (
            "prefill_forward",
            "decode_forward",
            "_attention_prefill",
            "_attention_decode",
            "_dense_mlp_decode",
            "_dense_mlp_prefill",
            "_sparse_moe_chunk",
            "_sparse_moe_prefill_chunk",
            "_dense_expert_moe_chunk",
        )
    )
    for forbidden in ("import torch", "from_torch", "to_torch", "untilize", "tilize"):
        assert forbidden not in runtime_source
    assert "sparse_matmul" in inspect.getsource(OptimizedDecoder._sparse_moe_chunk)
    assert "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig" in inspect.getsource(
        __import__(
            "models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder",
            fromlist=["_decode_dram_program"],
        )._decode_dram_program
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [1, 31, 33, 65])
def test_optimized_dense_non_aligned_prefill_matches_reference(mesh_device, seq_len):
    config = _config()
    state = _synthetic_state(config, 0)
    decoder = OptimizedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=0, mesh_device=mesh_device, batch=1, max_cache_len=96
    )
    generator = torch.Generator().manual_seed(22000 + seq_len)
    hidden = _randn(generator, 1, seq_len, config.hidden_size, scale=0.02)
    reference, _ = _dense_reference(hidden, torch.arange(seq_len).reshape(1, -1), state, config)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 3), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = decoder.build_rope_rows(torch.arange(seq_len), hf_config=config)
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    _assert_pcc(f"optimized-dense-prefill-{seq_len}", reference, ttnn.to_torch(actual).squeeze(0))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx,batch", [(0, 1), (0, 32), (1, 1), (1, 32), (4, 1)])
def test_optimized_traced_decode_layer_kinds_and_batches(mesh_device, layer_idx, batch):
    config = _config()
    state = _synthetic_state(config, layer_idx, sparse_weights=layer_idx != 0)
    kv_dtype = (
        ttnn.bfloat8_b if os.environ.get("NORTH_MINI_KV_DTYPE") == "bfp8" else OptimizationConfig().kv_cache_dtype
    )
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=32,
        optimization_config=OptimizationConfig(kv_cache_dtype=kv_dtype),
    )
    generator = torch.Generator().manual_seed(23000 + layer_idx + batch)
    hidden = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    reference = _position_zero_reference(hidden, state, config, layer_idx)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(batch, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0] * batch)
    hidden_tt = _to_tt(hidden.unsqueeze(0), mesh_device)
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
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc(
            f"optimized-layer-{layer_idx}-batch-{batch}-trace",
            reference,
            ttnn.to_torch(actual).squeeze(0),
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_decode_determinism_and_repeated_trace(mesh_device):
    config = _config()
    state = _synthetic_state(config, 1, sparse_weights=True)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=1,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=32,
        optimization_config=_expert_sweep_policy(),
    )
    generator = torch.Generator().manual_seed(24001)
    hidden_a = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    hidden_b = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0])
    hidden_tt = _to_tt(hidden_a.unsqueeze(0), mesh_device)
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
        outputs = []
        for _ in range(10):
            ttnn.copy_host_to_device_tensor(_to_host_tt(hidden_b.unsqueeze(0), mesh_device), hidden_tt)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            outputs.append(ttnn.to_torch(actual))
        assert all(torch.equal(outputs[0], output) for output in outputs[1:])
        reference = _position_zero_reference(hidden_b, state, config, 1)
        _assert_pcc("optimized-repeated-trace", reference, outputs[-1].squeeze(0))
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [1, 4])
def test_optimized_real_weight_moe_decode(mesh_device, layer_idx):
    config = _config()
    hidden = _real_hidden_at_layer(layer_idx, config)
    state = _real_layer_state(layer_idx)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=32,
        optimization_config=_expert_sweep_policy(),
    )
    reference = _real_moe_reference(hidden, state, config, layer_idx)
    actual = _decode_once(decoder, hidden, config, mesh_device)
    _assert_pcc(
        f"optimized-real-layer{layer_idx}-decode-{os.environ.get('NORTH_MINI_EXPERT_POLICY', 'selected')}",
        reference,
        ttnn.to_torch(actual).squeeze(0),
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,batch,mode",
    [(layer, batch, mode) for layer in (1, 4) for batch in (1, 32) for mode in ("prefill", "decode")],
)
def test_optimized_real_weight_moe_precision_matrix(mesh_device, monkeypatch, layer_idx, batch, mode):
    """Exercise selected mixed or explicitly requested dense-expert precision on authentic activations."""
    config = _config()
    sequence = 33 if mode == "prefill" else 34
    hidden = _real_hidden_at_layer(layer_idx, config, sequence=sequence, token_id=321)
    state = _real_layer_state(layer_idx)
    expert_dtype_name = os.environ.get("NORTH_MINI_AUTHENTIC_EXPERT_DTYPE", "selected")
    expert_dtype = {
        "bf16": ttnn.bfloat16,
        "bfp8": ttnn.bfloat8_b,
        "bfp4": ttnn.bfloat4_b,
    }.get(expert_dtype_name)
    expert_fidelity_name = os.environ.get("NORTH_MINI_AUTHENTIC_EXPERT_FIDELITY", "lofi")
    expert_fidelity = {
        "lofi": ttnn.MathFidelity.LoFi,
        "hifi2": ttnn.MathFidelity.HiFi2,
    }[expert_fidelity_name]
    policy_overrides = dict(
        expert_gate_up_fidelity=expert_fidelity,
        expert_down_fidelity=expert_fidelity,
        dense_expert_batch_threshold=1,
        # The default matrix proves the selected mixed topology. Explicit
        # dtype sweeps force dense execution so every row exercises the dense
        # precision named by NORTH_MINI_AUTHENTIC_EXPERT_DTYPE.
        batch1_prefill_active_experts=expert_dtype_name == "selected",
        dense_expert_chunk_size=int(os.environ.get("NORTH_MINI_AUTHENTIC_EXPERT_CHUNK", "1024")),
        serving_fused_kv_update=os.environ.get("NORTH_MINI_AUTHENTIC_FUSED_KV", "0") == "1",
    )
    if expert_dtype is not None:
        policy_overrides.update(
            dense_expert_gate_up_dtype=expert_dtype,
            dense_expert_down_dtype=expert_dtype,
        )
    policy = replace(OptimizationConfig(), **policy_overrides)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=64,
        optimization_config=policy,
    )
    if expert_dtype_name == "selected":
        assert decoder.weights["expert_gate"].dtype == ttnn.bfloat8_b
        assert decoder.weights["expert_down"].dtype == ttnn.bfloat8_b
        assert decoder.weights["dense_expert_gate"].dtype == ttnn.bfloat4_b
        assert decoder.weights["dense_expert_down"].dtype == ttnn.bfloat4_b
    branch_calls = {"dense": 0, "active_prefill": 0}
    original_dense = decoder._dense_expert_moe_chunk
    original_active_prefill = decoder._sparse_moe_prefill_chunk

    def counted_dense(*args, **kwargs):
        branch_calls["dense"] += 1
        return original_dense(*args, **kwargs)

    def counted_active_prefill(*args, **kwargs):
        branch_calls["active_prefill"] += 1
        return original_active_prefill(*args, **kwargs)

    monkeypatch.setattr(decoder, "_dense_expert_moe_chunk", counted_dense)
    monkeypatch.setattr(decoder, "_sparse_moe_prefill_chunk", counted_active_prefill)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(batch, 2), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    if mode == "prefill":
        reference = _real_moe_reference(hidden, state, config, layer_idx).repeat(batch, 1, 1)
        hidden = hidden.repeat(batch, 1, 1)
        cos, sin = decoder.build_rope_rows(torch.arange(sequence), hf_config=config)
        actual = decoder.prefill_forward(
            _to_tt(hidden.unsqueeze(0), mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            position_cos=_to_tt(cos, mesh_device) if decoder.use_rope else None,
            position_sin=_to_tt(sin, mesh_device) if decoder.use_rope else None,
        )
    else:
        prefix = hidden[:, :-1]
        decode_hidden = hidden[:, -1:]
        prefix_positions = torch.arange(sequence - 1).reshape(1, -1)
        _, reference_cache = _real_moe_reference(
            prefix,
            state,
            config,
            layer_idx,
            positions=prefix_positions,
            return_cache=True,
        )
        decode_positions = torch.full((1, 1), sequence - 1, dtype=torch.long)
        reference = _real_moe_reference(
            decode_hidden,
            state,
            config,
            layer_idx,
            positions=decode_positions,
            cache=reference_cache,
        ).repeat(batch, 1, 1)
        prefix = prefix.repeat(batch, 1, 1)
        decode_hidden = decode_hidden.repeat(batch, 1, 1)
        cos, sin = decoder.build_rope_rows(torch.arange(sequence - 1), hf_config=config)
        decoder.prefill_forward(
            _to_tt(prefix.unsqueeze(0), mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            position_cos=_to_tt(cos, mesh_device) if decoder.use_rope else None,
            position_sin=_to_tt(sin, mesh_device) if decoder.use_rope else None,
        )
        branch_calls["dense"] = 0
        branch_calls["active_prefill"] = 0
        hidden_tt = _to_tt(decode_hidden.unsqueeze(0), mesh_device)
        current, cos_tt, sin_tt = _decode_inputs(decoder, config, mesh_device, [sequence - 1] * batch)
        kwargs = dict(
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=current,
            position_cos=cos_tt if decoder.use_rope else None,
            position_sin=sin_tt if decoder.use_rope else None,
        )
        decoder.decode_forward(hidden_tt, **kwargs)
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        actual = decoder.decode_forward(hidden_tt, **kwargs)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        try:
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            actual_host = ttnn.to_torch(actual).squeeze(0)
        finally:
            ttnn.release_trace(mesh_device, trace_id)
    if mode == "prefill":
        actual_host = ttnn.to_torch(actual).squeeze(0)

    selected_active_prefill = expert_dtype_name == "selected" and batch == 1 and mode == "prefill"
    if selected_active_prefill:
        assert branch_calls["active_prefill"] > 0
        assert branch_calls["dense"] == 0
    else:
        assert branch_calls["dense"] > 0
        assert branch_calls["active_prefill"] == 0
    _assert_pcc(
        f"optimized-real-{expert_dtype_name}-layer{layer_idx}-{mode}-b{batch}",
        reference,
        actual_host,
    )


_DRAM_EXPERT_BLOCK_PAIRS = (
    (1, 1),
    (2, 2),
    (4, 3),
    (8, 4),
    (16, 6),
    (32, 8),
    (64, 12),
    (64, 24),
)
_DRAM_EXPERT_PCC_THRESHOLD = 0.995


def _dram_expert_block_pairs():
    value = os.environ.get("NORTH_MINI_DRAM_EXPERT_BLOCK_PAIRS")
    if value is None:
        return _DRAM_EXPERT_BLOCK_PAIRS
    pairs = []
    for item in value.split(","):
        gate_up, down = (int(field) for field in item.split(":", maxsplit=1))
        if 64 % gate_up or 24 % down:
            raise ValueError(
                "NORTH_MINI_DRAM_EXPERT_BLOCK_PAIRS entries must be gate_up:down pairs "
                "where gate_up divides 64 and down divides 24"
            )
        pairs.append((gate_up, down))
    if not pairs:
        raise ValueError("NORTH_MINI_DRAM_EXPERT_BLOCK_PAIRS cannot be empty")
    return tuple(dict.fromkeys(pairs))


def _dram_expert_selected(value, variable):
    selected = os.environ.get(variable)
    if selected is None:
        return True
    return str(value) in {item.strip() for item in selected.split(",")}


def _dram_expert_pcc(reference, actual):
    reference = reference.float().reshape(-1)
    actual = actual.float().reshape(-1)
    reference = reference - reference.mean()
    actual = actual - actual.mean()
    denominator = torch.linalg.vector_norm(reference) * torch.linalg.vector_norm(actual)
    if denominator == 0:
        return float(torch.equal(reference, actual))
    return float(torch.dot(reference, actual) / denominator)


def _dram_expert_is_capacity_failure(message):
    message = message.lower()
    return any(
        token in message
        for token in (
            "l1",
            "circular buffer",
            "cb allocation",
            "out of memory",
            "insufficient memory",
            "not enough space",
            "cannot allocate",
            "allocator",
        )
    )


def _dram_expert_memory_configs(mesh_device, group_count, k, n):
    workers = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_banks = len(workers)
    if group_count % num_banks:
        raise ValueError(f"group_count={group_count} must be divisible by num_dram_banks={num_banks}")
    worker_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(core.x, core.y), ttnn.CoreCoord(core.x, core.y)) for core in workers]
    )
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    groups_per_bank = group_count // num_banks
    input_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            worker_grid,
            [groups_per_bank * ttnn.TILE_SIZE, k],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    weight_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            dram_grid,
            [groups_per_bank * k, n],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    output_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            worker_grid,
            [groups_per_bank * ttnn.TILE_SIZE, n],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    return input_memory, weight_memory, output_memory, num_banks


def _real_expert_group(state, layer_idx, projection, start, end):
    prefix = f"model.layers.{layer_idx}.mlp.experts."
    return (
        torch.stack([state[f"{prefix}{expert}.{projection}_proj.weight"] for expert in range(start, end)])
        .transpose(-2, -1)
        .contiguous()
    )


def _materialize_dram_expert_groups(mesh_device, state, layer_idx, group_count, dtype, topology):
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    _, gate_weight_memory, _gate_output_memory, num_banks = _dram_expert_memory_configs(
        mesh_device, group_count, 2048, 768
    )
    _, packed_weight_memory, _packed_output_memory, _ = _dram_expert_memory_configs(
        mesh_device, group_count, 2048, 1536
    )
    _, down_weight_memory, _down_output_memory, _ = _dram_expert_memory_configs(mesh_device, group_count, 768, 2048)
    groups = []
    for start in range(0, 128, group_count):
        materialized = {}
        if topology == "packed":
            gate = _real_expert_group(state, layer_idx, "gate", start, start + group_count)
            up = _real_expert_group(state, layer_idx, "up", start, start + group_count)
            materialized["gate_up"] = ttnn.from_torch(
                torch.cat((gate, up), dim=-1).unsqueeze(0),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=packed_weight_memory,
                mesh_mapper=mapper,
            )
            del gate, up
            projections = ("down",)
        else:
            projections = ("gate", "up", "down")
        for projection in projections:
            host = _real_expert_group(state, layer_idx, projection, start, start + group_count)
            materialized[projection] = ttnn.from_torch(
                host.unsqueeze(0),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=gate_weight_memory if projection != "down" else down_weight_memory,
                mesh_mapper=mapper,
            )
            del host
        groups.append(materialized)
    return groups, num_banks


def _measure_warmed_expert_chain(mesh_device, operation, output_shape):
    warmups = int(os.environ.get("NORTH_MINI_DRAM_EXPERT_WARMUPS", "3"))
    iterations = int(os.environ.get("NORTH_MINI_DRAM_EXPERT_ITERATIONS", "20"))
    traced = os.environ.get("NORTH_MINI_DRAM_EXPERT_TRACE", "1") == "1"
    if warmups < 1 or iterations < 1:
        raise ValueError("NORTH_MINI_DRAM_EXPERT_WARMUPS and ITERATIONS must be positive")

    for _ in range(warmups):
        output = operation()
        ttnn.synchronize_device(mesh_device)
        output.deallocate(True)
        output = None

    if not traced:
        start = time.perf_counter_ns()
        for _ in range(iterations):
            output = operation()
        ttnn.synchronize_device(mesh_device)
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000 / iterations
        host = ttnn.to_torch(output).reshape(output_shape).float()
        output.deallocate(True)
        return latency_ms, host, "warmed_eager_full_chain"

    trace_id = None
    capture_ended = False
    try:
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output = operation()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        capture_ended = True
        for _ in range(warmups):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter_ns()
        for _ in range(iterations):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000 / iterations
        host = ttnn.to_torch(output).reshape(output_shape).float()
        return latency_ms, host, "warmed_trace_replay_full_chain"
    finally:
        if trace_id is not None and not capture_ended:
            try:
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            except Exception:
                pass
        if trace_id is not None:
            ttnn.release_trace(mesh_device, trace_id)
        if output is not None:
            output.deallocate(True)


def _dram_expert_candidate_operation(
    decoder,
    normalized,
    grouped_weights,
    group_count,
    gate_up_block_w,
    down_block_w,
    topology,
):
    mesh_device = decoder.mesh_device
    input_memory, _gate_weight_memory, gate_output_memory, _ = _dram_expert_memory_configs(
        mesh_device, group_count, decoder.hidden_size, decoder.intermediate_size
    )
    _packed_input_memory, _packed_weight_memory, packed_output_memory, _ = _dram_expert_memory_configs(
        mesh_device, group_count, decoder.hidden_size, 2 * decoder.intermediate_size
    )
    down_input_memory, _down_weight_memory, down_output_memory, _ = _dram_expert_memory_configs(
        mesh_device, group_count, decoder.intermediate_size, decoder.hidden_size
    )
    if down_input_memory != gate_output_memory:
        raise AssertionError("batched DRAM-sharded gate/up and down shard contracts do not compose")
    gate_program = ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
        in0_block_w=gate_up_block_w,
        per_core_M=1,
        per_core_N=(2 if topology == "packed" else 1) * decoder.intermediate_size // ttnn.TILE_SIZE,
        fused_activation=None,
    )
    down_program = ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
        in0_block_w=down_block_w,
        per_core_M=1,
        per_core_N=decoder.hidden_size // ttnn.TILE_SIZE,
        fused_activation=None,
    )

    def run():
        stage = "router"
        try:
            flat = ttnn.reshape(normalized, (decoder.batch, decoder.hidden_size))
            logits = ttnn.linear(
                flat,
                decoder.weights["router"],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(x=1, y=8),
                compute_kernel_config=decoder.router_compute,
            )
            top_values, top_indices = ttnn.topk(logits, k=decoder.top_k, dim=-1, sorted=True)
            top_values = ttnn.sigmoid(top_values)
            routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
            logits.deallocate(True)
            top_values.deallocate(True)
            top_indices.deallocate(True)
            routing = ttnn.permute(routing, (1, 0))
            routing = ttnn.reshape(routing, (1, decoder.num_experts, decoder.batch, 1))

            stage = "repeat_and_height_reshard"
            expert_input = ttnn.reshape(flat, (1, 1, decoder.batch, decoder.hidden_size))
            expert_input = ttnn.repeat(expert_input, ttnn.Shape((1, group_count, 1, 1)))
            expert_input = ttnn.to_memory_config(expert_input, input_memory)

            accumulator = None
            for group_index, weights in enumerate(grouped_weights):
                start = group_index * group_count
                if topology == "packed":
                    stage = f"group_{group_index}_packed_gate_up"
                    packed = ttnn.matmul(
                        expert_input,
                        weights["gate_up"],
                        dtype=ttnn.bfloat16,
                        memory_config=packed_output_memory,
                        program_config=gate_program,
                        compute_kernel_config=decoder.expert_gate_up_compute,
                    )
                    stage = f"group_{group_index}_packed_gate_slice"
                    gate = ttnn.slice(
                        packed,
                        (0, 0, 0, 0),
                        (1, group_count, decoder.batch, decoder.intermediate_size),
                        memory_config=gate_output_memory,
                    )
                    stage = f"group_{group_index}_packed_up_slice"
                    up = ttnn.slice(
                        packed,
                        (0, 0, 0, decoder.intermediate_size),
                        (1, group_count, decoder.batch, 2 * decoder.intermediate_size),
                        memory_config=gate_output_memory,
                    )
                    packed.deallocate(True)
                else:
                    stage = f"group_{group_index}_gate"
                    gate = ttnn.matmul(
                        expert_input,
                        weights["gate"],
                        dtype=ttnn.bfloat16,
                        memory_config=gate_output_memory,
                        program_config=gate_program,
                        compute_kernel_config=decoder.expert_gate_up_compute,
                    )
                    stage = f"group_{group_index}_up"
                    up = ttnn.matmul(
                        expert_input,
                        weights["up"],
                        dtype=ttnn.bfloat16,
                        memory_config=gate_output_memory,
                        program_config=gate_program,
                        compute_kernel_config=decoder.expert_gate_up_compute,
                    )
                stage = f"group_{group_index}_silu_multiply"
                activated = ttnn.multiply(
                    gate,
                    up,
                    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                    dtype=ttnn.bfloat16,
                    memory_config=gate_output_memory,
                )
                gate.deallocate(True)
                up.deallocate(True)

                stage = f"group_{group_index}_down"
                down = ttnn.matmul(
                    activated,
                    weights["down"],
                    dtype=ttnn.bfloat16,
                    memory_config=down_output_memory,
                    program_config=down_program,
                    compute_kernel_config=decoder.expert_down_compute,
                )
                activated.deallocate(True)

                stage = f"group_{group_index}_routing"
                routing_group = ttnn.slice(
                    routing,
                    (0, start, 0, 0),
                    (1, start + group_count, decoder.batch, 1),
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                routed = ttnn.multiply(
                    down,
                    routing_group,
                    dtype=ttnn.bfloat16,
                    memory_config=down_output_memory,
                )
                down.deallocate(True)
                routing_group.deallocate(True)

                stage = f"group_{group_index}_reduce"
                reduced = ttnn.experimental.fast_reduce_nc(
                    routed,
                    dims=[1],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                routed.deallocate(True)
                if accumulator is None:
                    accumulator = reduced
                else:
                    stage = f"group_{group_index}_accumulate"
                    previous = accumulator
                    accumulator = ttnn.add(previous, reduced, memory_config=ttnn.L1_MEMORY_CONFIG)
                    previous.deallocate(True)
                    reduced.deallocate(True)

            expert_input.deallocate(True)
            routing.deallocate(True)
            stage = "residual_boundary_reshard"
            accumulator = ttnn.to_memory_config(accumulator, decoder.residual_memory_config)
            return ttnn.reshape(accumulator, (1, 1, decoder.batch, decoder.hidden_size))
        except Exception as error:
            raise RuntimeError(f"DRAM expert full-chain failure at {stage}: {type(error).__name__}: {error}") from error

    return run


def _write_dram_expert_sweep_record(record):
    print("NORTH_MINI_DRAM_EXPERT_SWEEP_RESULT=" + json.dumps(record, sort_keys=True))
    output_dir = os.environ.get("NORTH_MINI_DRAM_EXPERT_SWEEP_OUTPUT_DIR")
    if output_dir is None:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    name = f"layer{record['layer_idx']}_g{record['group_count']}_" f"{record['weight_dtype']}_{record['topology']}.json"
    (output_path / name).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


@pytest.mark.skipif(
    os.environ.get("NORTH_MINI_DRAM_EXPERT_SWEEP", "0") != "1",
    reason="set NORTH_MINI_DRAM_EXPERT_SWEEP=1 for the real-weight all-expert DRAM-sharded sweep",
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [1, 4], ids=["layer1", "layer4"])
@pytest.mark.parametrize("group_count", [8, 16, 32, 64], ids=lambda value: f"g{value}")
@pytest.mark.parametrize("weight_dtype_name", ["bfp4", "bfp8"])
def test_dram_sharded_expert_full_chain_candidate(
    mesh_device,
    layer_idx,
    group_count,
    weight_dtype_name,
):
    """Execute every real expert through the compatible batched DRAM-sharded chain."""
    if not _dram_expert_selected(layer_idx, "NORTH_MINI_DRAM_EXPERT_LAYERS"):
        pytest.skip("layer filtered by NORTH_MINI_DRAM_EXPERT_LAYERS")
    if not _dram_expert_selected(group_count, "NORTH_MINI_DRAM_EXPERT_GROUPS"):
        pytest.skip("group filtered by NORTH_MINI_DRAM_EXPERT_GROUPS")
    if not _dram_expert_selected(weight_dtype_name, "NORTH_MINI_DRAM_EXPERT_DTYPES"):
        pytest.skip("dtype filtered by NORTH_MINI_DRAM_EXPERT_DTYPES")
    topology = os.environ.get("NORTH_MINI_DRAM_EXPERT_TOPOLOGY", "split")
    if topology not in {"split", "packed"}:
        raise ValueError("NORTH_MINI_DRAM_EXPERT_TOPOLOGY must be split or packed")
    if topology == "packed" and (group_count != 8 or weight_dtype_name != "bfp4"):
        pytest.skip("the scoped packed candidate is restricted to real G8 BFP4")

    config = _config()
    batch = 32
    state = _real_layer_state(layer_idx)
    hidden = _real_hidden_at_layer(layer_idx, config, sequence=34, token_id=321)
    normalized = _normalized(hidden[:, -1:], state, config, layer_idx).repeat(batch, 1, 1)
    policy = replace(
        OptimizationConfig(),
        dense_expert_batch_threshold=1,
        dense_expert_gate_up_dtype=ttnn.bfloat4_b,
        dense_expert_down_dtype=ttnn.bfloat4_b,
        expert_gate_up_fidelity=ttnn.MathFidelity.LoFi,
        expert_down_fidelity=ttnn.MathFidelity.LoFi,
    )
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=64,
        optimization_config=policy,
    )
    assert decoder.weights["dense_expert_gate"].dtype == ttnn.bfloat4_b
    assert decoder.weights["dense_expert_up"].dtype == ttnn.bfloat4_b
    assert decoder.weights["dense_expert_down"].dtype == ttnn.bfloat4_b
    normalized_tt = ttnn.from_torch(
        normalized.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    def baseline_operation():
        output = decoder._dense_expert_moe_chunk(normalized_tt, batch)
        output = ttnn.reshape(output, (1, 1, batch, config.hidden_size))
        return ttnn.to_memory_config(output, decoder.residual_memory_config)

    baseline_ms, baseline, timing_kind = _measure_warmed_expert_chain(
        mesh_device,
        baseline_operation,
        (batch, config.hidden_size),
    )
    weight_dtype = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}[weight_dtype_name]
    grouped_weights, num_banks = _materialize_dram_expert_groups(
        mesh_device,
        state,
        layer_idx,
        group_count,
        weight_dtype,
        topology,
    )
    assert len(grouped_weights) * group_count == config.num_experts
    record = {
        "workload": "real_weight_b32_dense_expert_full_chain",
        "layer_idx": layer_idx,
        "group_count": group_count,
        "groups_executed": 128 // group_count,
        "experts_executed": 128,
        "num_dram_banks": num_banks,
        "weight_dtype": weight_dtype_name,
        "topology": topology,
        "activation_dtype": "bfloat16",
        "math_fidelity": "LoFi",
        "input_shape": [1, group_count, 32, 2048],
        "gate_up_weight_shape": [1, group_count, 2048, 768],
        "packed_gate_up_weight_shape": [1, group_count, 2048, 1536] if topology == "packed" else None,
        "down_weight_shape": [1, group_count, 768, 2048],
        "gate_up_matmuls_per_group": 1 if topology == "packed" else 2,
        "gate_up_per_core_n": (1536 if topology == "packed" else 768) // ttnn.TILE_SIZE,
        "packed_split": (
            {
                "api": "two tile-aligned ttnn.slice calls",
                "ttnn_split_supported": False,
                "output_memory_layout": "HEIGHT_SHARDED_L1",
            }
            if topology == "packed"
            else None
        ),
        "baseline": {
            "path": "OptimizedDecoder._dense_expert_moe_chunk",
            "weight_dtype": "bfp4",
            "latency_ms": baseline_ms,
            "timing": timing_kind,
        },
        "candidates": [],
    }

    for gate_up_block_w, down_block_w in _dram_expert_block_pairs():
        row = {
            "topology": topology,
            "gate_up_in0_block_w": gate_up_block_w,
            "down_in0_block_w": down_block_w,
        }
        operation = _dram_expert_candidate_operation(
            decoder,
            normalized_tt,
            grouped_weights,
            group_count,
            gate_up_block_w,
            down_block_w,
            topology,
        )
        try:
            latency_ms, actual, candidate_timing_kind = _measure_warmed_expert_chain(
                mesh_device,
                operation,
                (batch, config.hidden_size),
            )
            pcc = _dram_expert_pcc(baseline, actual)
            row.update(
                {
                    "status": "correct" if pcc >= _DRAM_EXPERT_PCC_THRESHOLD else "pcc_rejected",
                    "pcc_vs_dense_bfp4": pcc,
                    "pcc_threshold": _DRAM_EXPERT_PCC_THRESHOLD,
                    "latency_ms": latency_ms,
                    "speedup_vs_dense_bfp4": baseline_ms / latency_ms,
                    "timing": candidate_timing_kind,
                }
            )
        except Exception as error:
            message = str(error)
            row.update(
                {
                    "status": "runtime_rejected",
                    "error_type": type(error).__name__,
                    "error": message,
                    "capacity_failure": _dram_expert_is_capacity_failure(message),
                }
            )
            gc.collect()
        record["candidates"].append(row)

    _write_dram_expert_sweep_record(record)
    normalized_tt.deallocate(True)
    for weights in grouped_weights:
        for weight in weights.values():
            weight.deallocate(True)
    correct = [row for row in record["candidates"] if row["status"] == "correct"]
    runtime_failures = [row for row in record["candidates"] if row["status"] == "runtime_rejected"]
    if group_count == 8 and weight_dtype_name == "bfp4" and topology == "split":
        assert correct, (
            f"no correct real-weight full-chain candidate survived for layer={layer_idx}, "
            f"group_count={group_count}, dtype={weight_dtype_name}: {record['candidates']}"
        )
    else:
        assert correct or any(
            row["status"] in {"pcc_rejected", "runtime_rejected"} for row in record["candidates"]
        ), f"group_count={group_count} produced neither a survivor nor rejection evidence"
        if group_count >= 32 and not correct:
            non_capacity = [row for row in runtime_failures if not row["capacity_failure"]]
            assert not non_capacity, (
                f"G{group_count} may only be rejected by exact L1/circular-buffer capacity evidence; "
                f"non-capacity failures: {non_capacity}"
            )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,batch,sequence",
    [(layer_idx, 1, sequence) for layer_idx in (1, 4) for sequence in (33, 128)],
)
def test_optimized_non_aligned_sparse_prefill_exercises_active_experts(
    mesh_device, monkeypatch, layer_idx, batch, sequence
):
    config = _config()
    total_tokens = batch * sequence
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    prefix = f"model.layers.{layer_idx}."
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        state[prefix + f"self_attn.{projection}.weight"].zero_()
    generator = torch.Generator().manual_seed(26000 + layer_idx)
    hidden = _randn(generator, batch, sequence, config.hidden_size, scale=0.02)
    normalized = _normalized(hidden, state, config, layer_idx)
    router_logits = F.linear(normalized.reshape(-1, config.hidden_size), state[prefix + "mlp.gate.weight"])
    router_top = torch.topk(router_logits, config.num_experts_per_tok + 1, dim=-1).values
    route_margin = router_top[:, -2] - router_top[:, -1]
    selected_tokens = []
    for anchor in (0, total_tokens // 2, total_tokens - 1):
        start, end = max(0, anchor - 8), min(total_tokens, anchor + 9)
        selected_tokens.append(start + int(route_margin[start:end].argmax()))
    selected_tokens = sorted(set(selected_tokens))
    reference_moe, experts = _sparse_moe_reference(
        normalized,
        state,
        config,
        layer_idx,
        flat_indices=selected_tokens,
    )
    assert torch.unique(experts).numel() > config.num_experts_per_tok
    reference = hidden.reshape(-1, config.hidden_size)[selected_tokens] + reference_moe
    policy = _expert_sweep_policy()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=max(64, sequence),
        optimization_config=policy,
    )
    sparse_chunks = []
    original_sparse = decoder._sparse_moe_prefill_chunk

    def counted_sparse(value, token_count):
        sparse_chunks.append(token_count)
        return original_sparse(value, token_count)

    def forbidden_decode_sparse(*_args, **_kwargs):
        pytest.fail("selected prefill entered token-as-batch decode sparse execution")

    def forbidden_dense(*_args, **_kwargs):
        pytest.fail("active-expert test entered dense all-expert execution")

    monkeypatch.setattr(decoder, "_sparse_moe_prefill_chunk", counted_sparse)
    monkeypatch.setattr(decoder, "_sparse_moe_chunk", forbidden_decode_sparse)
    monkeypatch.setattr(decoder, "_dense_expert_moe_chunk", forbidden_dense)
    actual_moe = decoder._sparse_moe(
        _to_tt(normalized.unsqueeze(0), mesh_device),
        sequence,
        phase="prefill",
    )
    actual_moe = ttnn.to_torch(actual_moe).squeeze(0).reshape(-1, config.hidden_size)[selected_tokens]
    _assert_pcc(f"optimized-active-experts-moe-layer-{layer_idx}", reference_moe, actual_moe)
    sparse_chunks.clear()
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(
        _page_table(batch, math.ceil(sequence / decoder.page_size)),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cos, sin = decoder.build_rope_rows(torch.arange(sequence), hf_config=config)
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device) if decoder.use_rope else None,
        position_sin=_to_tt(sin, mesh_device) if decoder.use_rope else None,
    )
    expected_chunks = [
        min(policy.prefill_moe_chunk_size, total_tokens - start)
        for start in range(0, total_tokens, policy.prefill_moe_chunk_size)
    ]
    assert sparse_chunks == expected_chunks
    actual = ttnn.to_torch(actual).squeeze(0).reshape(-1, config.hidden_size)[selected_tokens]
    for token, (reference_row, actual_row) in enumerate(zip(reference, actual)):
        _assert_pcc(
            f"optimized-active-experts-layer-{layer_idx}-sample-{token}",
            reference_row,
            actual_row,
            threshold=-1.0,
        )
    _assert_pcc(f"optimized-active-experts-layer-{layer_idx}", reference, actual)


@pytest.mark.parametrize("batch,sequence", [(32, 1), (2, 33)])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_serving_prefill_exercises_selected_dense_experts(mesh_device, monkeypatch, batch, sequence):
    config = _config()
    layer_idx = 1
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    prefix = f"model.layers.{layer_idx}."
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        state[prefix + f"self_attn.{projection}.weight"].zero_()
    router = state[prefix + "mlp.gate.weight"]
    router.fill_(-1.0)
    for expert in range(config.num_experts_per_tok):
        router[expert].fill_(1.0 - 0.05 * expert)
    generator = torch.Generator().manual_seed(27001)
    hidden = _randn(generator, batch, sequence, config.hidden_size, scale=0.02).abs() + 0.01
    normalized = _normalized(hidden, state, config, layer_idx)
    reference_moe, _ = _sparse_moe_reference(normalized, state, config, layer_idx)
    reference = hidden + reference_moe.reshape_as(hidden)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=64,
        optimization_config=_expert_sweep_policy(),
    )
    dense_calls = []
    original_dense = decoder._dense_expert_moe_chunk

    def counted_dense(value, token_count):
        dense_calls.append(token_count)
        return original_dense(value, token_count)

    def forbidden_sparse(*_args, **_kwargs):
        pytest.fail("selected serving prefill entered sparse expert execution")

    monkeypatch.setattr(decoder, "_dense_expert_moe_chunk", counted_dense)
    monkeypatch.setattr(decoder, "_sparse_moe_chunk", forbidden_sparse)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(batch, 2), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = decoder.build_rope_rows(torch.arange(sequence), hf_config=config)
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    assert dense_calls == [batch * sequence]
    actual_host = ttnn.to_torch(actual).squeeze(0)
    assert torch.isfinite(actual_host.float()).all()
    _assert_pcc(
        f"optimized-serving-prefill-dense-experts-synthetic-diagnostic-b{batch}-s{sequence}",
        reference,
        actual_host,
        threshold=-1.0,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_nonzero_positions_update_permuted_paged_cache(mesh_device):
    config = _config()
    state = _synthetic_state(config, 0)
    batch, positions = 4, [5, 17, 31, 63]
    decoder = OptimizedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=0, mesh_device=mesh_device, batch=batch, max_cache_len=64
    )
    generator = torch.Generator().manual_seed(26463)
    hidden = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    table = _page_table(batch, 2)
    page_table = _to_tt(table, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, positions)
    decoder.decode_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos,
        position_sin=sin,
    )
    prefix = "model.layers.0."
    normalized = _normalized(hidden, state, config, 0)
    expected_key = F.linear(normalized, state[prefix + "self_attn.k_proj.weight"])
    expected_key = expected_key.view(batch, 1, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    expected_key = _rope_interleaved(expected_key, torch.tensor(positions).reshape(batch, 1), config)
    expected_key = torch.cat((expected_key[..., ::2], expected_key[..., 1::2]), dim=-1)
    expected_value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    expected_value = expected_value.view(batch, 1, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    physical_key, physical_value = ttnn.to_torch(key_cache), ttnn.to_torch(value_cache)
    for user, position in enumerate(positions):
        block = int(table[user, position // decoder.page_size])
        slot = position % decoder.page_size
        _assert_pcc(
            f"optimized-key-slot-{user}",
            expected_key[user, :, 0],
            physical_key[block, :, slot],
        )
        _assert_pcc(
            f"optimized-value-slot-{user}",
            expected_value[user, :, 0],
            physical_value[block, :, slot],
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_optimized_real_weight_dense_precision_policy(mesh_device, mode):
    config = _config()
    state = _real_layer_zero_state()
    policy_name = os.environ.get("NORTH_MINI_PRECISION_POLICY", "selected")
    policies = {
        "selected": OptimizationConfig(),
        "all_bfp4_lofi": OptimizationConfig(
            attention_weight_dtype=ttnn.bfloat4_b,
            dense_gate_up_dtype=ttnn.bfloat4_b,
            dense_down_dtype=ttnn.bfloat4_b,
            prefill_dense_gate_up_dtype=ttnn.bfloat4_b,
            prefill_dense_down_dtype=ttnn.bfloat4_b,
            attention_fidelity=ttnn.MathFidelity.LoFi,
            dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
            dense_down_fidelity=ttnn.MathFidelity.LoFi,
            prefill_dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
            prefill_dense_down_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "bfp8_attention_bfp4_mlp_lofi": OptimizationConfig(
            dense_gate_up_dtype=ttnn.bfloat4_b,
            dense_down_dtype=ttnn.bfloat4_b,
            attention_fidelity=ttnn.MathFidelity.LoFi,
            dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
            dense_down_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "bfp8_attention_bfp4_mlp": OptimizationConfig(
            dense_gate_up_dtype=ttnn.bfloat4_b,
            dense_down_dtype=ttnn.bfloat4_b,
            dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
            dense_down_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "bfp4_gate_only": OptimizationConfig(
            dense_gate_up_dtype=ttnn.bfloat4_b,
            dense_gate_up_fidelity=ttnn.MathFidelity.LoFi,
        ),
    }
    if policy_name not in policies:
        raise ValueError(f"unknown NORTH_MINI_PRECISION_POLICY={policy_name!r}")
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=64,
        optimization_config=policies[policy_name],
    )
    sequence = 1 if mode == "decode" else 33
    hidden = _real_embedding_hidden(token_id=124, sequence=sequence)
    reference = _dense_reference(hidden, torch.arange(sequence).reshape(1, -1), state, config)[0]
    if mode == "decode":
        actual = _decode_once(decoder, hidden, config, mesh_device)
    else:
        key_cache, value_cache = decoder.create_paged_kv_cache()
        page_table = _to_tt(_page_table(1, 2), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        cos, sin = decoder.build_rope_rows(torch.arange(sequence), hf_config=config)
        actual = decoder.prefill_forward(
            _to_tt(hidden.unsqueeze(0), mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            position_cos=_to_tt(cos, mesh_device),
            position_sin=_to_tt(sin, mesh_device),
        )
    _assert_pcc(
        f"optimized-real-layer0-{mode}-{policy_name}",
        reference,
        ttnn.to_torch(actual).squeeze(0),
    )
