# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (
    _assert_pcc,
    _attention_reference,
    _config,
    _decode_inputs,
    _dense_reference,
    _normalized,
    _page_table,
    _randn,
    _real_layer_one_state,
    _rope_interleaved,
    _sparse_moe_reference,
    _synthetic_state,
    _to_host_tt,
    _to_tt,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import (
    ADVERTISED_CONTEXT,
    POLICIES,
    OptimizedDecoder,
)
from models.common.lightweightmodule import LightweightModule


def test_optimized_contract_and_no_functional_runtime_fallback():
    assert issubclass(OptimizedDecoder, LightweightModule)
    assert ADVERTISED_CONTEXT == _config().max_position_embeddings
    assert {"default", "bf16_reference", "bfp8_hifi2", "bfp8_lofi", "bfp4_attention", "bf16_cache"} <= set(POLICIES)
    prefill = inspect.signature(OptimizedDecoder.prefill_forward)
    decode = inspect.signature(OptimizedDecoder.decode_forward)
    assert {"key_cache", "value_cache", "page_table", "position_cos", "position_sin"} <= set(prefill.parameters)
    assert {"page_table", "current_positions", "position_cos", "position_sin"} <= set(decode.parameters)
    runtime_methods = (
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder.decode_forward,
        OptimizedDecoder._attention_prefill,
        OptimizedDecoder._attention_decode,
        OptimizedDecoder._dense_mlp,
        OptimizedDecoder._routing,
        OptimizedDecoder._sparse_decode_moe,
        OptimizedDecoder._sparse_prefill_moe,
        OptimizedDecoder._dense_expert_moe,
        OptimizedDecoder._mlp,
    )
    runtime_source = "\n".join(inspect.getsource(method) for method in runtime_methods)
    for forbidden in ("FunctionalDecoder", "import torch", "from_torch", "to_torch"):
        assert forbidden not in runtime_source
    assert "sparse_matmul" in inspect.getsource(OptimizedDecoder._sparse_decode_moe)
    assert "repeat(expert_input" not in inspect.getsource(OptimizedDecoder._sparse_decode_moe)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [31, 33])
@pytest.mark.parametrize("candidate", ["bf16_reference", "bfp8_hifi2", "default", "large_prefill"])
def test_optimized_dense_non_aligned_prefill(mesh_device, seq_len, candidate):
    config = _config()
    state = _synthetic_state(config, 0)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=64,
        candidate=candidate,
    )
    generator = torch.Generator().manual_seed(27000 + seq_len)
    hidden = _randn(generator, 1, seq_len, config.hidden_size, scale=0.02).unsqueeze(0)
    reference, _ = _dense_reference(hidden.squeeze(0), torch.arange(seq_len).reshape(1, -1), state, config)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 2), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = decoder.build_rope_rows(torch.arange(seq_len), hf_config=config)
    actual = decoder.prefill_forward(
        _to_tt(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    _assert_pcc(f"optimized-{candidate}-dense-prefill-{seq_len}", reference, ttnn.to_torch(actual).squeeze(0))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("candidate", ["bf16_reference", "bfp8_hifi2", "default"])
def test_optimized_dense_paged_decode_trace(mesh_device, candidate):
    config = _config()
    state = _synthetic_state(config, 0)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=64,
        candidate=candidate,
    )
    generator = torch.Generator().manual_seed(28001)
    prefill_len = 33
    prefill_hidden = _randn(generator, 1, prefill_len, config.hidden_size, scale=0.02)
    _, reference_cache = _dense_reference(prefill_hidden, torch.arange(prefill_len).reshape(1, -1), state, config)
    decode_hidden = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    reference, _ = _dense_reference(decode_hidden, torch.tensor([[prefill_len]]), state, config, cache=reference_cache)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 2), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
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
    kwargs = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "page_table": page_table,
        "current_positions": current,
        "position_cos": cos_tt,
        "position_sin": sin_tt,
    }
    decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    actual = decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc(f"optimized-{candidate}-dense-traced-decode", reference, ttnn.to_torch(actual).squeeze(0))
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "candidate,batch",
    [
        ("bfp8_hifi2", 1),
        ("default", 1),
        ("geometry_12x30", 1),
        ("geometry_24x24", 1),
        ("packed_sparse_gate_up", 1),
        ("default", 32),
        ("packed_interleaved_48_64", 32),
        ("dram_sharded_attention_bfp8", 1),
    ],
)
def test_optimized_moe_decode_candidate_matches_reference(mesh_device, candidate, batch):
    config = _config()
    layer_idx = 4
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=32,
        candidate=candidate,
    )
    generator = torch.Generator().manual_seed(29000)
    hidden = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(batch, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, _, _ = _decode_inputs(decoder, config, mesh_device, [0] * batch)
    actual = decoder.decode_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
    )
    ttnn.synchronize_device(mesh_device)

    prefix = f"model.layers.{layer_idx}."
    normalized = _normalized(hidden, state, config, layer_idx)
    value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    value = value.view(batch, 1, config.num_key_value_heads, config.head_dim)
    attention = value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=2).reshape(
        batch, 1, -1
    )
    attention = F.linear(attention, state[prefix + "self_attn.o_proj.weight"])
    moe, _ = _sparse_moe_reference(normalized, state, config, layer_idx)
    reference = hidden + attention + moe.reshape_as(hidden)
    _assert_pcc(
        f"optimized-moe-decode-{candidate}-batch-{batch}",
        reference,
        ttnn.to_torch(actual).squeeze(0),
        threshold=0.995,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [1, 4])
@pytest.mark.parametrize("seq_len", [31, 33])
def test_optimized_moe_non_aligned_prefill_matches_reference(mesh_device, layer_idx, seq_len):
    config = _config()
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=64,
    )
    generator = torch.Generator().manual_seed(30000 + 100 * layer_idx + seq_len)
    hidden = _randn(generator, 1, seq_len, config.hidden_size, scale=0.02)
    positions = torch.arange(seq_len).reshape(1, -1)
    attention, _ = _attention_reference(hidden, positions, state, config, layer_idx)
    normalized = _normalized(hidden, state, config, layer_idx)
    moe, _ = _sparse_moe_reference(normalized, state, config, layer_idx)
    reference = attention + moe.reshape_as(hidden)

    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 2), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = decoder.build_rope_rows(torch.arange(seq_len), hf_config=config)
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device) if decoder.use_rope else None,
        position_sin=_to_tt(sin, mesh_device) if decoder.use_rope else None,
    )
    _assert_pcc(
        f"optimized-moe-layer-{layer_idx}-prefill-{seq_len}",
        reference,
        ttnn.to_torch(actual).squeeze(0),
        threshold=0.995,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_nonzero_positions_physical_cache_and_determinism(mesh_device):
    config = _config()
    state = _synthetic_state(config, 0)
    batch, positions = 4, [5, 17, 31, 63]
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=64,
    )
    generator = torch.Generator().manual_seed(31000)
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
    normalized = _normalized(hidden, state, config, 0)
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
        _assert_pcc(f"optimized-key-slot-{user}", expected_key[user, :, 0], physical_key[physical_block, :, slot])
        _assert_pcc(f"optimized-value-slot-{user}", expected_value[user, :, 0], physical_value[physical_block, :, slot])


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_real_weight_sliding_moe_decode(mesh_device):
    config = _config()
    state = _real_layer_one_state()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=1,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=32,
    )
    generator = torch.Generator().manual_seed(123)
    hidden = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    prefix = "model.layers.1."
    normalized = _normalized(hidden, state, config, 1)
    value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    value = value.view(1, 1, config.num_key_value_heads, config.head_dim)
    attention = value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=2).reshape(
        1, 1, -1
    )
    attention = F.linear(attention, state[prefix + "self_attn.o_proj.weight"])
    flat = normalized.reshape(1, -1)
    logits = F.linear(flat, state[prefix + "mlp.gate.weight"])
    scores, experts = torch.topk(logits, config.num_experts_per_tok, dim=-1)
    scores = torch.sigmoid(scores)
    moe = torch.zeros_like(flat)
    for route, expert in enumerate(experts[0].tolist()):
        gate = F.linear(flat, state[f"{prefix}mlp.experts.{expert}.gate_proj.weight"])
        up = F.linear(flat, state[f"{prefix}mlp.experts.{expert}.up_proj.weight"])
        contribution = F.linear(
            F.silu(gate) * up,
            state[f"{prefix}mlp.experts.{expert}.down_proj.weight"],
        )
        moe += contribution * scores[0, route]
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
    _assert_pcc("optimized-real-layer1-decode", reference, ttnn.to_torch(actual).squeeze(0), threshold=0.995)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx,batch", [(1, 32), (4, 1)])
def test_optimized_moe_dynamic_trace_replay(mesh_device, layer_idx, batch):
    config = _config()
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=32,
    )
    generator = torch.Generator().manual_seed(32000 + layer_idx + batch)
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
            f"optimized-moe-layer-{layer_idx}-batch-{batch}-dynamic-trace",
            reference,
            ttnn.to_torch(actual).squeeze(0),
            threshold=0.995,
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)
