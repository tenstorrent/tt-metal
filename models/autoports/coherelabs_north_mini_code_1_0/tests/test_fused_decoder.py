# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import inspect
import math

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
    _rope_interleaved,
    _sparse_moe_reference,
    _synthetic_state,
    _to_host_tt,
    _to_tt,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import FunctionalDecoder
from models.autoports.coherelabs_north_mini_code_1_0.tt.fused_decoder import FusedDecoder


def test_fused_path_is_materially_overridden_and_runtime_clean():
    assert issubclass(FusedDecoder, FunctionalDecoder)
    assert FusedDecoder._dense_mlp is not FunctionalDecoder._dense_mlp
    assert FusedDecoder._attention_decode is not FunctionalDecoder._attention_decode
    assert FusedDecoder._sparse_moe_chunk is not FunctionalDecoder._sparse_moe_chunk
    dense_source = inspect.getsource(FusedDecoder._dense_mlp)
    sparse_source = inspect.getsource(FusedDecoder._sparse_moe_chunk)
    swiglu_source = inspect.getsource(FusedDecoder._swiglu)
    assert 'self.weights["gate_up"]' in dense_source
    assert 'self.weights["expert_gate_up"]' in sparse_source
    assert "normalized.shape[2] == 1" in dense_source
    assert "input_tensor_a_activations" in swiglu_source
    defaults = inspect.signature(FusedDecoder.from_state_dict).parameters
    assert defaults["dense_gate_up_variant"].default == "packed_slice"
    assert defaults["sparse_gate_up_variant"].default == "packed"
    runtime_source = "\n".join(
        inspect.getsource(method)
        for method in (
            FusedDecoder._swiglu,
            FusedDecoder._dense_mlp,
            FusedDecoder._sparse_moe_chunk,
            FusedDecoder.prefill_forward,
            FusedDecoder.decode_forward,
        )
    )
    for forbidden in ("torch.", "from_torch", "to_torch"):
        assert forbidden not in runtime_source


@pytest.mark.parametrize("seq_len", [33, 65])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fused_dense_non_aligned_prefill_matches_reference(mesh_device, seq_len):
    config = _config()
    state = _synthetic_state(config, 0)
    decoder = FusedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=0, mesh_device=mesh_device, max_cache_len=96
    )
    assert "gate_up" in decoder.weights
    assert "gate_proj" not in decoder.weights
    assert "up_proj" not in decoder.weights
    hidden = torch.randn(1, seq_len, config.hidden_size, generator=torch.Generator().manual_seed(2026)) * 0.01
    positions = torch.arange(seq_len)
    cos, sin = decoder.build_rope_rows(positions, hf_config=config)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 3), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    reference, _ = _dense_reference(hidden, positions.unsqueeze(0), state, config)
    _assert_pcc("fused-dense-prefill", reference, ttnn.to_torch(actual).squeeze(0), threshold=0.995)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fused_dense_paged_decode_trace_replay_matches_reference(mesh_device):
    config = _config()
    state = _synthetic_state(config, 0)
    decoder = FusedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=0, mesh_device=mesh_device, max_cache_len=64
    )
    hidden = torch.randn(1, 1, config.hidden_size, generator=torch.Generator().manual_seed(7)) * 0.01
    positions = torch.tensor([0])
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 2), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    hidden_tt = _to_tt(hidden.unsqueeze(0), mesh_device)
    pos_tt, cos_tt, sin_tt = _decode_inputs(decoder, config, mesh_device, positions)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
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
        _assert_pcc("fused-dense-traced-decode", reference, ttnn.to_torch(actual).squeeze(0), threshold=0.995)
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("layer_idx", [1, 4])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fused_sparse_layer_kinds_prefill(mesh_device, layer_idx):
    config = _config()
    state = _synthetic_state(config, layer_idx)
    decoder = FusedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=layer_idx, mesh_device=mesh_device, max_cache_len=64
    )
    assert "expert_gate_up" in decoder.weights
    assert "expert_gate" not in decoder.weights
    assert "expert_up" not in decoder.weights
    hidden = torch.zeros(1, 33, config.hidden_size)
    positions = torch.arange(33)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 2), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    kwargs = {}
    if decoder.use_rope:
        cos, sin = decoder.build_rope_rows(positions, hf_config=config)
        kwargs.update(position_cos=_to_tt(cos, mesh_device), position_sin=_to_tt(sin, mesh_device))
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        **kwargs,
    )
    reference, _ = _attention_reference(hidden, positions.unsqueeze(0), state, config, layer_idx)
    _assert_pcc(f"fused-layer-{layer_idx}-prefill", reference, ttnn.to_torch(actual).squeeze(0), threshold=0.995)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx,batch", [(1, 32), (4, 1)])
def test_fused_sparse_dynamic_trace_replay_matches_reference(mesh_device, layer_idx, batch):
    config = _config()
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    decoder = FusedDecoder.from_state_dict(
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
            f"fused-sparse-layer-{layer_idx}-batch-{batch}-trace",
            reference,
            ttnn.to_torch(actual).squeeze(0),
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [1, 4])
def test_packed_sparse_nonzero_prefill_matches_active_experts(mesh_device, layer_idx):
    config = _config()
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    prefix = f"model.layers.{layer_idx}."
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        state[prefix + f"self_attn.{projection}.weight"].zero_()
    sequence, selected_tokens = 33, [0, 16, 32]
    hidden = _randn(torch.Generator().manual_seed(16000 + layer_idx), 1, sequence, config.hidden_size, scale=0.02)
    normalized = _normalized(hidden, state, config, layer_idx)
    reference_moe, _ = _sparse_moe_reference(normalized, state, config, layer_idx, flat_indices=selected_tokens)
    reference = hidden[:, selected_tokens] + reference_moe.reshape(1, len(selected_tokens), -1)
    decoder = FusedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=sequence,
        sparse_gate_up_variant="packed",
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(
        _page_table(1, math.ceil(sequence / decoder.page_size)),
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
    actual = ttnn.to_torch(actual).squeeze(0)[:, selected_tokens]
    _assert_pcc(f"packed-sparse-layer-{layer_idx}-prefill", reference, actual)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fused_paged_cache_slots_and_decode_determinism(mesh_device):
    config = _config()
    state = _synthetic_state(config, 0)
    batch, positions = 4, [5, 17, 31, 63]
    decoder = FusedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=0, mesh_device=mesh_device, batch=batch, max_cache_len=64
    )
    hidden = _randn(torch.Generator().manual_seed(9463), batch, 1, config.hidden_size, scale=0.02)
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
        _assert_pcc(f"fused-key-slot-{user}", expected_key[user, :, 0], physical_key[physical_block, :, slot])
        _assert_pcc(f"fused-value-slot-{user}", expected_value[user, :, 0], physical_value[physical_block, :, slot])
