# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimized-path ownership and correctness coverage."""

import inspect
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
    _page_table,
    _real_layer_one_state,
    _synthetic_state,
    _to_tt,
)
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_fused_decoder import (
    _normalized,
    _randn,
    _rope_interleaved,
    _sparse_moe_reference,
    _to_host_tt,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import OptimizedDecoder


def _real_dense_state_and_activations():
    snapshot = Path(f"/huggingface/hub/models--CohereLabs--North-Mini-Code-1.0/snapshots/{REAL_REVISION}")
    shard = snapshot / "model-00001-of-00049.safetensors"
    if not shard.is_file():
        pytest.skip("North-Mini checkpoint shard 1 is not cached")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        state = {key: handle.get_tensor(key) for key in handle.keys() if key.startswith("model.layers.0.")}
        embeddings = handle.get_tensor("model.embed_tokens.weight")[:65].unsqueeze(0)
    return state, embeddings


def test_optimized_path_owns_dense_runtime():
    assert OptimizedDecoder._dense_mlp.__qualname__.startswith("OptimizedDecoder.")
    source = inspect.getsource(OptimizedDecoder._dense_mlp)
    assert "compute_kernel_config=self.mlp_compute_kernel_config" in source
    assert "FunctionalDecoder" not in source


@pytest.mark.parametrize("seq_len", [33, 65])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_dense_non_aligned_prefill(mesh_device, seq_len):
    config = _config()
    state = _synthetic_state(config, 0)
    decoder = OptimizedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=0, mesh_device=mesh_device, max_cache_len=96
    )
    hidden = torch.randn(1, seq_len, config.hidden_size, generator=torch.Generator().manual_seed(2026)) * 0.01
    positions = torch.arange(seq_len)
    cos, sin = decoder.build_rope_rows(positions, hf_config=config)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_to_tt(_page_table(1, 3), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    reference, _ = _dense_reference(hidden, positions.unsqueeze(0), state, config)
    _assert_pcc("optimized-dense-prefill", reference, ttnn.to_torch(actual).squeeze(0), threshold=0.995)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_dense_traced_decode(mesh_device):
    config = _config()
    state = _synthetic_state(config, 0)
    decoder = OptimizedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=0, mesh_device=mesh_device, max_cache_len=64
    )
    hidden = torch.randn(1, 1, config.hidden_size, generator=torch.Generator().manual_seed(7)) * 0.01
    positions = torch.tensor([0])
    key_cache, value_cache = decoder.create_paged_kv_cache()
    hidden_tt = _to_tt(hidden.unsqueeze(0), mesh_device)
    pos_tt, cos_tt, sin_tt = _decode_inputs(decoder, config, mesh_device, positions)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_to_tt(_page_table(1, 2), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        current_positions=pos_tt,
        position_cos=cos_tt,
        position_sin=sin_tt,
    )
    decoder.decode_forward(hidden_tt, **kwargs)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    actual = decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        reference, _ = _dense_reference(hidden, positions.unsqueeze(0), state, config)
        _assert_pcc("optimized-dense-decode", reference, ttnn.to_torch(actual).squeeze(0), threshold=0.995)
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("layer_idx", [1, 4])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_sparse_layer_kinds_prefill(mesh_device, layer_idx):
    config = _config()
    state = _synthetic_state(config, layer_idx)
    decoder = OptimizedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=layer_idx, mesh_device=mesh_device, max_cache_len=64
    )
    hidden = torch.zeros(1, 33, config.hidden_size)
    positions = torch.arange(33)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    kwargs = {}
    if decoder.use_rope:
        cos, sin = decoder.build_rope_rows(positions, hf_config=config)
        kwargs.update(position_cos=_to_tt(cos, mesh_device), position_sin=_to_tt(sin, mesh_device))
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_to_tt(_page_table(1, 2), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        **kwargs,
    )
    reference, _ = _attention_reference(hidden, positions.unsqueeze(0), state, config, layer_idx)
    _assert_pcc(f"optimized-layer-{layer_idx}-prefill", reference, ttnn.to_torch(actual).squeeze(0), threshold=0.995)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx,batch", [(1, 32), (4, 1)])
def test_optimized_sparse_dynamic_trace_replay(mesh_device, layer_idx, batch):
    config = _config()
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=32,
        sparse_gate_up_variant="packed",
    )
    generator = torch.Generator().manual_seed(18000 + layer_idx + batch)
    hidden_a = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    hidden_b = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    hidden_tt = _to_tt(hidden_a.unsqueeze(0), mesh_device)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0] * batch)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_to_tt(_page_table(batch, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
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
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        prefix = f"model.layers.{layer_idx}."
        normalized = _normalized(hidden_b, state, config, layer_idx)
        value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
        value = value.view(batch, 1, config.num_key_value_heads, config.head_dim)
        attention = value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=2)
        attention = F.linear(attention.reshape(batch, 1, -1), state[prefix + "self_attn.o_proj.weight"])
        moe, _ = _sparse_moe_reference(normalized, state, config, layer_idx)
        _assert_pcc(
            f"optimized-sparse-{layer_idx}-batch-{batch}-trace",
            hidden_b + attention + moe.reshape_as(hidden_b),
            ttnn.to_torch(actual).squeeze(0),
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_paged_cache_slots_and_determinism(mesh_device):
    config = _config()
    state = _synthetic_state(config, 0)
    batch, positions = 4, [5, 17, 31, 63]
    decoder = OptimizedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=0, mesh_device=mesh_device, batch=batch, max_cache_len=64
    )
    hidden = _randn(torch.Generator().manual_seed(9463), batch, 1, config.hidden_size, scale=0.02)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    table = _page_table(batch, 2)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, positions)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_to_tt(table, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
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
    physical_key, physical_value = ttnn.to_torch(key_cache), ttnn.to_torch(value_cache)
    for user, position in enumerate(positions):
        block, slot = int(table[user, position // decoder.page_size]), position % decoder.page_size
        _assert_pcc(f"optimized-key-slot-{user}", expected_key[user, :, 0], physical_key[block, :, slot])
        _assert_pcc(f"optimized-value-slot-{user}", expected_value[user, :, 0], physical_value[block, :, slot])


@pytest.mark.parametrize("candidate", ["bfp8_hifi2", "bfp4_attention", "bfp4_mlp", "all_bfp4"])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_real_weight_recorded_activation_dense_prefill(mesh_device, candidate):
    """Precision gate using checkpoint weights and actual token embeddings."""
    config = _config()
    state, hidden = _real_dense_state_and_activations()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_cache_len=65,
        candidate=candidate,
    )
    positions = torch.arange(65)
    cos, sin = decoder.build_rope_rows(positions, hf_config=config)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_to_tt(_page_table(1, 3), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    reference, _ = _dense_reference(hidden, positions.unsqueeze(0), state, config)
    _assert_pcc(f"real-dense-prefill-{candidate}", reference, ttnn.to_torch(actual).squeeze(0), threshold=0.995)


@pytest.mark.parametrize("candidate", ["bfp8_hifi2"])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_real_weight_recorded_activation_sparse_decode(mesh_device, candidate):
    config = _config()
    state = _real_layer_one_state()
    _, embedding_rows = _real_dense_state_and_activations()
    hidden = embedding_rows[:, :1]
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=1,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=32,
        candidate=candidate,
    )
    prefix = "model.layers.1."
    normalized = _normalized(hidden, state, config, 1)
    value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    value = value.view(1, 1, config.num_key_value_heads, config.head_dim)
    attention = value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=2)
    attention = F.linear(attention.reshape(1, 1, -1), state[prefix + "self_attn.o_proj.weight"])
    flat = normalized.reshape(1, -1)
    logits = F.linear(flat, state[prefix + "mlp.gate.weight"])
    scores, experts = torch.topk(logits, config.num_experts_per_tok, dim=-1)
    scores = torch.sigmoid(scores)
    moe = torch.zeros_like(flat)
    for rank, expert in enumerate(experts[0].tolist()):
        gate = F.linear(flat, state[f"{prefix}mlp.experts.{expert}.gate_proj.weight"])
        up = F.linear(flat, state[f"{prefix}mlp.experts.{expert}.up_proj.weight"])
        down = F.linear(F.silu(gate) * up, state[f"{prefix}mlp.experts.{expert}.down_proj.weight"])
        moe += down * scores[0, rank]
    reference = hidden + attention + moe.reshape_as(hidden)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0])
    actual = decoder.decode_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_to_tt(_page_table(1, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        current_positions=current,
        position_cos=cos,
        position_sin=sin,
    )
    _assert_pcc(f"real-sparse-decode-{candidate}", reference, ttnn.to_torch(actual).squeeze(0), threshold=0.995)
