# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import pytest
import torch
from transformers.cache_utils import StaticCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_functional_decoder import (
    DECODE_CACHE_POSITION,
    EMITTED_BATCH_SIZE,
    EMITTED_DECODE_CACHE_LEN,
    LARGE_SEQ_LEN,
    NON_ALIGNED_SEQ_LEN,
    _assert_pcc,
    _layer_local_state_dict,
    _load_real_model_or_skip,
    _run_reference_prefill,
    _synthetic_state_dict,
    _tt_tensor,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_functional_decoder import (  # noqa: F401
    hf_config as hf_config,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_functional_decoder import (  # noqa: F401
    mesh_device as mesh_device,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    build_decode_mask,
    build_rope_tables,
    build_update_indices,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (
    OptimizedDecoder,
    OptimizedDecoderPolicy,
)

SYNTHETIC_STABLE_POLICY = OptimizedDecoderPolicy(
    projection_weight_dtype=ttnn.bfloat8_b,
    mlp_weight_dtype=ttnn.bfloat8_b,
)


def _hf_static_prefill_mask(seq_len, cache_len):
    mask = torch.full((1, 1, seq_len, cache_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    for row in range(seq_len):
        mask[:, :, row, : row + 1] = 0.0
    return mask


def _hf_static_decode_mask(cache_position, cache_len):
    mask = torch.full((1, 1, 1, cache_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    mask[:, :, :, : cache_position + 1] = 0.0
    return mask


def _run_reference_decode_for_layer(layer, rotary_emb, hf_config, layer_idx, prefix_hidden, decode_hidden):
    cache = StaticCache(config=hf_config, max_cache_len=EMITTED_DECODE_CACHE_LEN)
    cache.early_initialization(
        batch_size=EMITTED_BATCH_SIZE,
        num_heads=hf_config.num_key_value_heads,
        head_dim=hf_config.head_dim,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    prefix_len = prefix_hidden.shape[1]
    prefix_positions = torch.arange(prefix_len, dtype=torch.long)
    with torch.no_grad():
        layer(
            prefix_hidden,
            attention_mask=_hf_static_prefill_mask(prefix_len, EMITTED_DECODE_CACHE_LEN),
            position_ids=prefix_positions.unsqueeze(0),
            position_embeddings=rotary_emb(prefix_hidden, prefix_positions.unsqueeze(0)),
            past_key_values=cache,
            use_cache=True,
            cache_position=prefix_positions,
        )
        decode_position = torch.tensor([prefix_len], dtype=torch.long)
        prefix_key_cache = cache.layers[layer_idx].keys.clone()
        prefix_value_cache = cache.layers[layer_idx].values.clone()
        reference = layer(
            decode_hidden,
            attention_mask=_hf_static_decode_mask(prefix_len, EMITTED_DECODE_CACHE_LEN),
            position_ids=decode_position.unsqueeze(0),
            position_embeddings=rotary_emb(decode_hidden, decode_position.unsqueeze(0)),
            past_key_values=cache,
            use_cache=True,
            cache_position=decode_position,
        )
    return (
        reference,
        prefix_key_cache,
        prefix_value_cache,
        cache.layers[layer_idx].keys[:, :, prefix_len : prefix_len + 1, :].clone(),
        cache.layers[layer_idx].values[:, :, prefix_len : prefix_len + 1, :].clone(),
    )


def _assert_decode_cache_slot_updated(key_cache, value_cache, expected_key_update, expected_value_update):
    key_cache_torch = ttnn.to_torch(key_cache).to(torch.float32)
    value_cache_torch = ttnn.to_torch(value_cache).to(torch.float32)
    actual_key_update = key_cache_torch[:, :, DECODE_CACHE_POSITION : DECODE_CACHE_POSITION + 1, :]
    actual_value_update = value_cache_torch[:, :, DECODE_CACHE_POSITION : DECODE_CACHE_POSITION + 1, :]
    _assert_pcc(expected_key_update.to(torch.float32), actual_key_update, 0.99)
    _assert_pcc(expected_value_update.to(torch.float32), actual_value_update, 0.99)
    print("paged_cache_update_slot_checked=key,value")


def _optimized_prefill(decoder, hidden_states, mesh):
    tt_input = _tt_tensor(
        hidden_states.reshape(1, EMITTED_BATCH_SIZE, hidden_states.shape[1], hidden_states.shape[2]), mesh
    )
    return ttnn.to_torch(decoder.prefill_forward(tt_input)).reshape_as(hidden_states).to(torch.float32)


@pytest.mark.parametrize("layer_idx,seq_len", [(0, NON_ALIGNED_SEQ_LEN), (31, NON_ALIGNED_SEQ_LEN), (0, LARGE_SEQ_LEN)])
def test_optimized_synthetic_prefill_matches_hf_and_covers_qkv_layer_kinds(hf_config, mesh_device, layer_idx, seq_len):
    layer = LlamaDecoderLayer(hf_config, layer_idx=layer_idx).to(dtype=torch.bfloat16).eval()
    state_dict = _synthetic_state_dict(hf_config, layer_idx=layer_idx)
    layer.load_state_dict(_layer_local_state_dict(state_dict, layer_idx=layer_idx))
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    torch.manual_seed(20260720 + layer_idx)
    hidden_states = torch.randn(EMITTED_BATCH_SIZE, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH_SIZE,
        max_seq_len=LARGE_SEQ_LEN,
        policy=SYNTHETIC_STABLE_POLICY,
    )

    assert isinstance(decoder, OptimizedDecoder)
    reference = _run_reference_prefill(layer, rotary_emb, hidden_states, seq_len)
    actual = _optimized_prefill(decoder, hidden_states, mesh_device)
    _assert_pcc(reference, actual, 0.99)


@pytest.mark.parametrize("layer_idx", [0, 31])
def test_optimized_synthetic_decode_matches_hf_paged_cache(hf_config, mesh_device, layer_idx):
    layer = LlamaDecoderLayer(hf_config, layer_idx=layer_idx).to(dtype=torch.bfloat16).eval()
    state_dict = _synthetic_state_dict(hf_config, layer_idx=layer_idx)
    layer.load_state_dict(_layer_local_state_dict(state_dict, layer_idx=layer_idx))
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    torch.manual_seed(20260721)
    prefix_hidden = torch.randn(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, hf_config.hidden_size, dtype=torch.bfloat16)
    decode_hidden = torch.randn(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size, dtype=torch.bfloat16)
    reference, key_cache, value_cache, expected_key_update, expected_value_update = _run_reference_decode_for_layer(
        layer, rotary_emb, hf_config, layer_idx, prefix_hidden, decode_hidden
    )

    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH_SIZE,
        max_seq_len=EMITTED_DECODE_CACHE_LEN,
        policy=SYNTHETIC_STABLE_POLICY,
    )
    position_cos, position_sin = build_rope_tables(
        hf_config,
        1,
        mesh_device,
        start_pos=DECODE_CACHE_POSITION,
    )
    tt_key_cache = _tt_tensor(key_cache, mesh_device)
    tt_value_cache = _tt_tensor(value_cache, mesh_device)
    tt_output = decoder.decode_forward(
        _tt_tensor(decode_hidden.reshape(1, EMITTED_BATCH_SIZE, 1, hf_config.hidden_size), mesh_device),
        key_cache=tt_key_cache,
        value_cache=tt_value_cache,
        update_idxs_tensor=build_update_indices(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, mesh_device),
        position_cos=position_cos,
        position_sin=position_sin,
        attention_mask=build_decode_mask(
            EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, EMITTED_DECODE_CACHE_LEN, mesh_device
        ),
    )
    actual = ttnn.to_torch(tt_output).reshape_as(reference).to(torch.float32)
    _assert_pcc(reference, actual, 0.99)
    _assert_decode_cache_slot_updated(tt_key_cache, tt_value_cache, expected_key_update, expected_value_update)


@pytest.mark.parametrize("layer_idx", [0, 31])
def test_optimized_decode_trace_replay_smoke(hf_config, mesh_device, layer_idx):
    layer = LlamaDecoderLayer(hf_config, layer_idx=layer_idx).to(dtype=torch.bfloat16).eval()
    state_dict = _synthetic_state_dict(hf_config, layer_idx=layer_idx)
    layer.load_state_dict(_layer_local_state_dict(state_dict, layer_idx=layer_idx))
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    torch.manual_seed(20260724)
    prefix_hidden = torch.randn(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, hf_config.hidden_size, dtype=torch.bfloat16)
    decode_hidden = torch.randn(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size, dtype=torch.bfloat16)
    reference, key_cache, value_cache, expected_key_update, expected_value_update = _run_reference_decode_for_layer(
        layer, rotary_emb, hf_config, layer_idx, prefix_hidden, decode_hidden
    )

    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH_SIZE,
        max_seq_len=EMITTED_DECODE_CACHE_LEN,
        policy=SYNTHETIC_STABLE_POLICY,
    )
    position_cos, position_sin = build_rope_tables(hf_config, 1, mesh_device, start_pos=DECODE_CACHE_POSITION)
    tt_input = _tt_tensor(decode_hidden.reshape(1, EMITTED_BATCH_SIZE, 1, hf_config.hidden_size), mesh_device)
    tt_key_cache = _tt_tensor(key_cache, mesh_device)
    tt_value_cache = _tt_tensor(value_cache, mesh_device)
    update_idxs = build_update_indices(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, mesh_device)
    attention_mask = build_decode_mask(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, EMITTED_DECODE_CACHE_LEN, mesh_device)

    decoder.decode_forward(
        tt_input,
        key_cache=tt_key_cache,
        value_cache=tt_value_cache,
        update_idxs_tensor=update_idxs,
        position_cos=position_cos,
        position_sin=position_sin,
        attention_mask=attention_mask,
    )
    ttnn.synchronize_device(mesh_device)
    _assert_decode_cache_slot_updated(tt_key_cache, tt_value_cache, expected_key_update, expected_value_update)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        tt_input,
        key_cache=tt_key_cache,
        value_cache=tt_value_cache,
        update_idxs_tensor=update_idxs,
        position_cos=position_cos,
        position_sin=position_sin,
        attention_mask=attention_mask,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.tracy_message("PERF_DECODE")
    for _ in range(3):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
    ttnn.tracy_message("PERF_DECODE_END")

    actual = ttnn.to_torch(traced_output).reshape_as(reference).to(torch.float32)
    _assert_pcc(reference, actual, 0.99)


def test_optimized_decode_trace_replay_is_deterministic(hf_config, mesh_device):
    layer_idx = 31
    layer = LlamaDecoderLayer(hf_config, layer_idx=layer_idx).to(dtype=torch.bfloat16).eval()
    state_dict = _synthetic_state_dict(hf_config, layer_idx=layer_idx)
    layer.load_state_dict(_layer_local_state_dict(state_dict, layer_idx=layer_idx))
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    torch.manual_seed(20260732)
    prefix_hidden = torch.randn(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, hf_config.hidden_size, dtype=torch.bfloat16)
    decode_hidden = torch.randn(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size, dtype=torch.bfloat16)
    reference, key_cache, value_cache, expected_key_update, expected_value_update = _run_reference_decode_for_layer(
        layer, rotary_emb, hf_config, layer_idx, prefix_hidden, decode_hidden
    )

    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH_SIZE,
        max_seq_len=EMITTED_DECODE_CACHE_LEN,
        policy=SYNTHETIC_STABLE_POLICY,
    )
    position_cos, position_sin = build_rope_tables(hf_config, 1, mesh_device, start_pos=DECODE_CACHE_POSITION)
    tt_input = _tt_tensor(decode_hidden.reshape(1, EMITTED_BATCH_SIZE, 1, hf_config.hidden_size), mesh_device)
    tt_key_cache = _tt_tensor(key_cache, mesh_device)
    tt_value_cache = _tt_tensor(value_cache, mesh_device)
    update_idxs = build_update_indices(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, mesh_device)
    attention_mask = build_decode_mask(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, EMITTED_DECODE_CACHE_LEN, mesh_device)

    decoder.decode_forward(
        tt_input,
        key_cache=tt_key_cache,
        value_cache=tt_value_cache,
        update_idxs_tensor=update_idxs,
        position_cos=position_cos,
        position_sin=position_sin,
        attention_mask=attention_mask,
    )
    ttnn.synchronize_device(mesh_device)
    _assert_decode_cache_slot_updated(tt_key_cache, tt_value_cache, expected_key_update, expected_value_update)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        tt_input,
        key_cache=tt_key_cache,
        value_cache=tt_value_cache,
        update_idxs_tensor=update_idxs,
        position_cos=position_cos,
        position_sin=position_sin,
        attention_mask=attention_mask,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    outputs = []
    for _ in range(3):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
        outputs.append(ttnn.to_torch(traced_output).reshape_as(reference).to(torch.float32))

    _assert_pcc(reference, outputs[-1], 0.99)
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[1], outputs[2])
    print("deterministic_replay_equal_rounds=3")


def test_optimized_signposted_prefill_perf_smoke(hf_config, mesh_device):
    model = _load_real_model_or_skip()
    hf_config = model.config
    state_dict = model.state_dict()
    torch.manual_seed(20260725)
    hidden_states = torch.randn(EMITTED_BATCH_SIZE, NON_ALIGNED_SEQ_LEN, hf_config.hidden_size, dtype=torch.bfloat16)
    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH_SIZE,
        max_seq_len=LARGE_SEQ_LEN,
    )

    reference = _run_reference_prefill(
        model.model.layers[0], model.model.rotary_emb, hidden_states, NON_ALIGNED_SEQ_LEN
    )
    tt_input = _tt_tensor(
        hidden_states.reshape(1, EMITTED_BATCH_SIZE, NON_ALIGNED_SEQ_LEN, hf_config.hidden_size), mesh_device
    )
    ttnn.tracy_message("PERF_PREFILL")
    tt_output = decoder.prefill_forward(tt_input)
    ttnn.synchronize_device(mesh_device)
    ttnn.tracy_message("PERF_PREFILL_END")

    actual = ttnn.to_torch(tt_output).reshape_as(reference).to(torch.float32)
    _assert_pcc(reference, actual, 0.99)


def test_optimized_real_weight_decode_trace_replay_if_weights_available():
    model = _load_real_model_or_skip()
    hf_config = model.config
    state_dict = model.state_dict()
    torch.manual_seed(20260730)
    prefix_hidden = torch.randn(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, hf_config.hidden_size, dtype=torch.bfloat16)
    decode_hidden = torch.randn(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size, dtype=torch.bfloat16)
    reference, key_cache, value_cache, expected_key_update, expected_value_update = _run_reference_decode_for_layer(
        model.model.layers[0],
        model.model.rotary_emb,
        hf_config,
        0,
        prefix_hidden,
        decode_hidden,
    )

    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh,
            batch=EMITTED_BATCH_SIZE,
            max_seq_len=EMITTED_DECODE_CACHE_LEN,
        )
        position_cos, position_sin = build_rope_tables(hf_config, 1, mesh, start_pos=DECODE_CACHE_POSITION)
        tt_input = _tt_tensor(decode_hidden.reshape(1, EMITTED_BATCH_SIZE, 1, hf_config.hidden_size), mesh)
        tt_key_cache = _tt_tensor(key_cache, mesh)
        tt_value_cache = _tt_tensor(value_cache, mesh)
        update_idxs = build_update_indices(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, mesh)
        attention_mask = build_decode_mask(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, EMITTED_DECODE_CACHE_LEN, mesh)

        decoder.decode_forward(
            tt_input,
            key_cache=tt_key_cache,
            value_cache=tt_value_cache,
            update_idxs_tensor=update_idxs,
            position_cos=position_cos,
            position_sin=position_sin,
            attention_mask=attention_mask,
        )
        ttnn.synchronize_device(mesh)
        _assert_decode_cache_slot_updated(tt_key_cache, tt_value_cache, expected_key_update, expected_value_update)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced_output = decoder.decode_forward(
            tt_input,
            key_cache=tt_key_cache,
            value_cache=tt_value_cache,
            update_idxs_tensor=update_idxs,
            position_cos=position_cos,
            position_sin=position_sin,
            attention_mask=attention_mask,
        )
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.tracy_message("PERF_DECODE")
        for _ in range(3):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
        ttnn.tracy_message("PERF_DECODE_END")

        actual = ttnn.to_torch(traced_output).reshape_as(reference).to(torch.float32)
        _assert_pcc(reference, actual, 0.99)
    finally:
        ttnn.close_mesh_device(mesh)


def test_optimized_real_weight_single_layer_prefill_matches_hf_if_weights_available():
    seq_len = NON_ALIGNED_SEQ_LEN
    model = _load_real_model_or_skip()
    hf_config = model.config
    state_dict = model.state_dict()
    rotary_emb = model.model.rotary_emb
    torch.manual_seed(20260722)
    hidden_states = torch.randn(EMITTED_BATCH_SIZE, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh,
            batch=EMITTED_BATCH_SIZE,
            max_seq_len=LARGE_SEQ_LEN,
        )
        reference = _run_reference_prefill(model.model.layers[0], rotary_emb, hidden_states, seq_len)
        actual = _optimized_prefill(decoder, hidden_states, mesh)
        _assert_pcc(reference, actual, 0.99)
    finally:
        ttnn.close_mesh_device(mesh)


def test_optimized_real_weight_single_layer_decode_matches_hf_if_weights_available():
    model = _load_real_model_or_skip()
    hf_config = model.config
    state_dict = model.state_dict()
    torch.manual_seed(20260727)
    prefix_hidden = torch.randn(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, hf_config.hidden_size, dtype=torch.bfloat16)
    decode_hidden = torch.randn(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size, dtype=torch.bfloat16)
    reference, key_cache, value_cache, expected_key_update, expected_value_update = _run_reference_decode_for_layer(
        model.model.layers[0],
        model.model.rotary_emb,
        hf_config,
        0,
        prefix_hidden,
        decode_hidden,
    )

    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh,
            batch=EMITTED_BATCH_SIZE,
            max_seq_len=EMITTED_DECODE_CACHE_LEN,
        )
        tt_key_cache = _tt_tensor(key_cache, mesh)
        tt_value_cache = _tt_tensor(value_cache, mesh)
        position_cos, position_sin = build_rope_tables(hf_config, 1, mesh, start_pos=DECODE_CACHE_POSITION)
        tt_output = decoder.decode_forward(
            _tt_tensor(decode_hidden.reshape(1, EMITTED_BATCH_SIZE, 1, hf_config.hidden_size), mesh),
            key_cache=tt_key_cache,
            value_cache=tt_value_cache,
            update_idxs_tensor=build_update_indices(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, mesh),
            position_cos=position_cos,
            position_sin=position_sin,
            attention_mask=build_decode_mask(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, EMITTED_DECODE_CACHE_LEN, mesh),
        )
        actual = ttnn.to_torch(tt_output).reshape_as(reference).to(torch.float32)
        _assert_pcc(reference, actual, 0.99)
        _assert_decode_cache_slot_updated(tt_key_cache, tt_value_cache, expected_key_update, expected_value_update)
    finally:
        ttnn.close_mesh_device(mesh)


def test_optimized_runtime_forwards_have_no_host_fallback():
    forbidden = ("torch", "from_torch", "to_torch", "FunctionalDecoder.forward")
    methods = [
        OptimizedDecoder._matmul,
        OptimizedDecoder._prefill_matmul,
        OptimizedDecoder._decode_dram_sharded_matmul,
        OptimizedDecoder._to_l1_width_sharded,
        OptimizedDecoder._attention_mlp,
        OptimizedDecoder._decode_attention_mlp,
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder.decode_forward,
        OptimizedDecoder.forward,
    ]
    hits = []
    for method in methods:
        source = inspect.getsource(method)
        hits.extend(f"{method.__name__}:{term}" for term in forbidden if term in source)
    assert hits == []
