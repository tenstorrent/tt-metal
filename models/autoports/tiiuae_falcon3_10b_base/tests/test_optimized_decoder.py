# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import inspect
import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open
from transformers import DynamicCache

import ttnn
from models.autoports.tiiuae_falcon3_10b_base.tests.test_functional_decoder import (
    REAL_WEIGHT_PCC,
    SYNTHETIC_PCC,
    _assert_pcc,
    _config,
    _hf_decode,
    _hf_decode_query,
    _hf_layer,
    _hf_prefill,
    _real_layer_state_dict,
    _synthetic_state_dict,
)
from models.autoports.tiiuae_falcon3_10b_base.tt.functional_decoder import (
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    IR_REPRESENTATIVE_LAYER,
    FunctionalDecoder,
)
from models.autoports.tiiuae_falcon3_10b_base.tt.optimized_decoder import OptimizedDecoder
from models.common.utility_functions import comp_pcc

ACTIVATION_FIXTURE = (
    Path(__file__).parents[1] / "doc" / "optimized_decoder" / "activations" / "layer20_inputs.safetensors"
)


def _assert_public_output_layout(tensor) -> None:
    assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG, tensor.memory_config()


def _assert_dram_decode_weight_contract(model) -> None:
    """Guard the prefill/decode weight-layout boundary found by AutoFix."""
    prefill_names = ("qkv_weight", "o_weight", "gate_weight", "up_weight", "gate_up_weight", "down_weight")
    decode_names = (
        "qkv_decode_weight",
        "o_decode_weight",
        "gate_decode_weight",
        "up_decode_weight",
        "gate_up_decode_weight",
        "down_decode_weight",
    )
    for name in prefill_names:
        weight = getattr(model, name, None)
        if weight is not None:
            assert weight.memory_config() == ttnn.DRAM_MEMORY_CONFIG, (name, weight.memory_config())
    for name in decode_names:
        weight = getattr(model, name, None)
        if weight is not None:
            memory_config = weight.memory_config()
            assert memory_config.buffer_type == ttnn.BufferType.DRAM, (name, memory_config)
            assert memory_config.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED, (name, memory_config)


def _write_result_artifact(filename: str, payload: dict) -> None:
    output_dir = os.getenv("FALCON3_RESULTS_DIR")
    if output_dir is None:
        return
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _recorded_layer20_inputs(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load recorded HF layer-20 inputs and expand the one-prompt fixture."""
    with safe_open(ACTIVATION_FIXTURE, framework="pt", device="cpu") as tensors:
        prefill = tensors.get_tensor("prefill")
        decode = tensors.get_tensor("decode")
    assert tuple(prefill.shape) == (1, EMITTED_PREFILL_SEQUENCE, 3072), prefill.shape
    assert tuple(decode.shape) == (1, 1, 3072), decode.shape
    return prefill.repeat(batch, 1, 1), decode.repeat(batch, 1, 1)


def _recorded_layer20_seq31_inputs(batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a genuine non-aligned prompt plus two cache-consuming tokens."""
    with safe_open(ACTIVATION_FIXTURE, framework="pt", device="cpu") as tensors:
        prefill = tensors.get_tensor("prefill_31")
        decode_31 = tensors.get_tensor("decode_31")
        decode_32 = tensors.get_tensor("decode_32")
    assert tuple(prefill.shape) == (1, 31, 3072), prefill.shape
    assert tuple(decode_31.shape) == (1, 1, 3072), decode_31.shape
    assert tuple(decode_32.shape) == (1, 1, 3072), decode_32.shape
    return tuple(tensor.repeat(batch, 1, 1) for tensor in (prefill, decode_31, decode_32))


def _recorded_layer20_seq128_input(batch: int) -> torch.Tensor:
    """Load a genuine max-fixture prompt propagated through HF layers 0..19."""
    with safe_open(ACTIVATION_FIXTURE, framework="pt", device="cpu") as tensors:
        prefill = tensors.get_tensor("prefill_128")
    assert tuple(prefill.shape) == (1, EMITTED_CACHE_LENGTH, 3072), prefill.shape
    return prefill.repeat(batch, 1, 1)


def _tt_input(hidden_states: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        hidden_states.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _cache_position(batch: int, position: int, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.full((batch,), position, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _device_cache_from_hf(model, hf_key, hf_value, device):
    """Materialize a decode-only cache fixture without invoking the batch-1 functional prefill alias bug."""
    key = torch.zeros((model.batch, model.num_kv_heads, model.max_cache_len, model.head_dim), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    key[:, :, : hf_key.shape[2], :] = hf_key
    value[:, :, : hf_value.shape[2], :] = hf_value
    dtype = getattr(model, "kv_cache_dtype", ttnn.bfloat16)
    return (
        ttnn.from_torch(
            key,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        ttnn.from_torch(
            value,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )


def _assert_recorded_cache_prefix_matches_hf(hf_cache, key_cache, value_cache, seq_len: int) -> None:
    hf_layer_cache = hf_cache.layers[IR_REPRESENTATIVE_LAYER]
    assert hf_layer_cache.is_initialized
    tt_key = ttnn.to_torch(key_cache)[:, :, :seq_len, :]
    tt_value = ttnn.to_torch(value_cache)[:, :, :seq_len, :]
    _assert_pcc("recorded layer-20 key cache", hf_layer_cache.keys, tt_key, REAL_WEIGHT_PCC)
    _assert_pcc("recorded layer-20 value cache", hf_layer_cache.values, tt_value, REAL_WEIGHT_PCC)


def _release_model(model) -> None:
    names = (
        "qkv_weight",
        "qkv_decode_weight",
        "o_weight",
        "o_decode_weight",
        "gate_weight",
        "up_weight",
        "gate_up_weight",
        "gate_decode_weight",
        "up_decode_weight",
        "gate_up_decode_weight",
        "down_weight",
        "down_decode_weight",
        "input_norm_weight",
        "post_attention_norm_weight",
        "cos_cache",
        "sin_cache",
        "decode_positions",
    )
    for name in names:
        tensor = getattr(model, name, None)
        if tensor is not None:
            tensor.deallocate(True)
    del model
    gc.collect()


def _run_prefill(model, device, hidden_states):
    key_cache, value_cache = model.allocate_kv_cache()
    tt_hidden = _tt_input(hidden_states, device)
    tt_output = model.prefill_forward(tt_hidden, key_cache=key_cache, value_cache=value_cache)
    _assert_public_output_layout(tt_output)
    output = ttnn.to_torch(tt_output).squeeze(0)
    tt_hidden.deallocate(True)
    tt_output.deallocate(True)
    return output, key_cache, value_cache


def _run_decode(model, device, hidden_states, key_cache, value_cache, position: int):
    tt_hidden = _tt_input(hidden_states, device)
    cache_position = _cache_position(model.batch, position, device)
    tt_output = model.decode_forward(
        tt_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=cache_position,
        position_index=position,
    )
    _assert_public_output_layout(tt_output)
    output = ttnn.to_torch(tt_output).squeeze(0)
    tt_hidden.deallocate(True)
    cache_position.deallocate(True)
    tt_output.deallocate(True)
    return output


def _run_decode_query(model, device, hidden_states, position: int):
    tt_hidden = _tt_input(hidden_states, device)
    residual = ttnn.reshape(tt_hidden, (1, 1, model.batch, model.hidden_size))
    residual = ttnn.to_memory_config(residual, model.residual_memory_config)
    query, key, value = model._decode_qkv(residual, position_index=position)
    output = ttnn.to_torch(query).squeeze(0)
    ttnn.deallocate(query)
    ttnn.deallocate(key)
    ttnn.deallocate(value)
    ttnn.deallocate(residual, force=False)
    ttnn.deallocate(tt_hidden)
    return output


def test_optimized_runtime_is_implementation_owned_and_host_fallback_free():
    hot_methods = (
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder.decode_forward,
        OptimizedDecoder._prefill_attention,
        OptimizedDecoder._decode_attention,
        OptimizedDecoder._decode_qkv,
        OptimizedDecoder._prefill_mlp,
        OptimizedDecoder._prefill_mlp_chunk,
        OptimizedDecoder._decode_mlp,
        OptimizedDecoder._decode_norm,
        OptimizedDecoder._move_owned,
        OptimizedDecoder._rotary_slice,
        OptimizedDecoder._unpad_prefill_sequence,
        OptimizedDecoder._prepare_decode_heads,
        OptimizedDecoder._decode_attention_mask,
    )
    forbidden = ("torch", "from_torch", "to_torch", "FunctionalDecoder.")
    for method in hot_methods:
        assert method.__qualname__.startswith("OptimizedDecoder."), method.__qualname__
        source = inspect.getsource(method)
        for token in forbidden:
            assert token not in source, f"{method.__name__} contains forbidden runtime token {token!r}"


@pytest.mark.use_module_device
@pytest.mark.timeout(1800)
def test_optimized_bf16_layout_prefill_decode_smoke(device):
    config = _config()
    batch = 1
    state_dict = _synthetic_state_dict(config, IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    model = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=device,
        batch=batch,
        precision_policy="bf16_hifi4",
        prefill_grid_x=8,
        prefill_in0_block_w=1,
    )

    generator = torch.Generator().manual_seed(41)
    seq_len = EMITTED_PREFILL_SEQUENCE
    hidden = torch.randn((batch, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    hf_cache = DynamicCache(config=config)
    expected_prefill = _hf_prefill(hf_layer, config, hidden, cache=hf_cache)
    actual_prefill, key_cache, value_cache = _run_prefill(model, device, hidden)
    _assert_pcc("optimized bf16 prefill seq=17", expected_prefill, actual_prefill, SYNTHETIC_PCC)

    decode_hidden = torch.randn((batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    expected_decode = _hf_decode(hf_layer, config, decode_hidden, hf_cache, seq_len)
    actual_decode = _run_decode(model, device, decode_hidden, key_cache, value_cache, seq_len)
    _assert_pcc("optimized bf16 decode position=17", expected_decode, actual_decode, SYNTHETIC_PCC)
    key_cache.deallocate(True)
    value_cache.deallocate(True)
    _release_model(model)


@pytest.mark.use_module_device
@pytest.mark.timeout(1800)
def test_shard_advisor_candidate_prefill_decode_smoke(device):
    config = _config()
    batch = 32
    state_dict = _synthetic_state_dict(config, IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    model = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=device,
        batch=batch,
        precision_policy="bfp8_hifi2",
        decode_matmul_mode="shard_advisor",
        prefill_grid_x=8,
        prefill_in0_block_w=1,
    )

    generator = torch.Generator().manual_seed(52)
    seq_len = EMITTED_PREFILL_SEQUENCE
    hidden = torch.randn((batch, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    hf_cache = DynamicCache(config=config)
    expected_prefill = _hf_prefill(hf_layer, config, hidden, cache=hf_cache)
    actual_prefill, key_cache, value_cache = _run_prefill(model, device, hidden)
    _assert_pcc("shard-advisor bfp8 prefill seq=17", expected_prefill, actual_prefill, SYNTHETIC_PCC)

    decode_hidden = torch.randn((batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    expected_decode = _hf_decode(hf_layer, config, decode_hidden, hf_cache, seq_len)
    actual_decode = _run_decode(model, device, decode_hidden, key_cache, value_cache, seq_len)
    _assert_pcc("shard-advisor bfp8 decode position=17", expected_decode, actual_decode, SYNTHETIC_PCC)
    key_cache.deallocate(True)
    value_cache.deallocate(True)
    _release_model(model)


@pytest.mark.use_module_device
@pytest.mark.timeout(1800)
def test_selected_decoder_semantics_cache_and_repeated_decode(device):
    """Re-run the functional contract on the selected optimized defaults."""
    config = _config()
    batch = 32
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    model = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=device,
        batch=batch,
    )
    assert model.precision_policy_name == "all_bfp4_lofi"
    assert model.decode_matmul_mode == "dram_sharded"
    assert model.dram_mlp_target_cores == 24
    _assert_dram_decode_weight_contract(model)
    assert model.down_input_memory_config == model.mlp_gate_output_memory_config

    hidden_17, _ = _recorded_layer20_inputs(batch)
    hidden_31, decode_31, decode_32 = _recorded_layer20_seq31_inputs(batch)
    hidden_128 = _recorded_layer20_seq128_input(batch)
    for hidden in (hidden_17, hidden_31, hidden_128):
        seq_len = hidden.shape[1]
        expected = _hf_prefill(hf_layer, config, hidden)
        actual, key_cache, value_cache = _run_prefill(model, device, hidden)
        _assert_pcc(f"selected recorded prefill seq={seq_len}", expected, actual, REAL_WEIGHT_PCC)
        ttnn.deallocate(key_cache)
        ttnn.deallocate(value_cache)

    past_len = 31
    hf_cache = DynamicCache(config=config)
    _hf_prefill(hf_layer, config, hidden_31, cache=hf_cache)
    _, key_cache, value_cache = _run_prefill(model, device, hidden_31)
    _assert_recorded_cache_prefix_matches_hf(hf_cache, key_cache, value_cache, past_len)

    for position, decode_hidden in ((past_len, decode_31), (past_len + 1, decode_32)):
        expected_query = _hf_decode_query(hf_layer, config, decode_hidden, position)
        actual_query = _run_decode_query(model, device, decode_hidden, position)
        _assert_pcc(
            f"selected recorded decode query position={position}", expected_query, actual_query, REAL_WEIGHT_PCC
        )
        expected = _hf_decode(hf_layer, config, decode_hidden, hf_cache, position)
        actual = _run_decode(model, device, decode_hidden, key_cache, value_cache, position)
        _assert_pcc(f"selected recorded decode position={position}", expected, actual, REAL_WEIGHT_PCC)
        _assert_recorded_cache_prefix_matches_hf(hf_cache, key_cache, value_cache, position + 1)

    # Rewriting the same cache slot is deterministic and exercises repeated dispatch.
    repeated = [_run_decode(model, device, decode_32, key_cache, value_cache, past_len + 1) for _ in range(8)]
    for output in repeated[1:]:
        assert torch.equal(repeated[0], output), "selected repeated decode is not bitwise deterministic"

    ttnn.deallocate(key_cache)
    ttnn.deallocate(value_cache)
    _release_model(model)


@pytest.mark.use_module_device
@pytest.mark.timeout(1800)
def test_selected_decoder_real_layer_prefill_decode(device):
    config = _config()
    batch = 32
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    model = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=device,
        batch=batch,
    )

    seq_len = EMITTED_PREFILL_SEQUENCE
    hidden, decode_hidden = _recorded_layer20_inputs(batch)
    hf_cache = DynamicCache(config=config)
    expected_prefill = _hf_prefill(hf_layer, config, hidden, cache=hf_cache)
    actual_prefill, key_cache, value_cache = _run_prefill(model, device, hidden)
    _assert_pcc("selected real layer 20 prefill", expected_prefill, actual_prefill, REAL_WEIGHT_PCC)

    expected_decode = _hf_decode(hf_layer, config, decode_hidden, hf_cache, seq_len)
    actual_decode = _run_decode(model, device, decode_hidden, key_cache, value_cache, seq_len)
    _assert_pcc("selected real layer 20 decode", expected_decode, actual_decode, REAL_WEIGHT_PCC)
    ttnn.deallocate(key_cache)
    ttnn.deallocate(value_cache)
    _release_model(model)


@pytest.mark.skipif(os.getenv("FALCON3_RUN_CAPACITY") != "1", reason="manual optimized context-capacity gate")
@pytest.mark.use_module_device
@pytest.mark.timeout(1800)
def test_selected_decoder_context_capacity(device):
    """Prove that optimization preserves the repo's advertised batch-32 context."""
    config = _config()
    batch = 32
    seq_len = int(os.getenv("FALCON3_CAPACITY_SEQ_LEN", "6528"))
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    model = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=device,
        batch=batch,
        max_cache_len=seq_len,
    )
    key_cache, value_cache = model.allocate_kv_cache()
    hidden = ttnn.zeros(
        (1, batch, seq_len, config.hidden_size),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = model.prefill_forward(hidden, key_cache=key_cache, value_cache=value_cache)
    ttnn.synchronize_device(device)
    assert tuple(int(value) for value in output.shape) == (1, batch, seq_len, config.hidden_size)
    _assert_public_output_layout(output)
    for tensor in (output, hidden, key_cache, value_cache):
        ttnn.deallocate(tensor)
    _release_model(model)


def _capture_and_time_decode(model, device, tt_hidden, key_cache, value_cache, cache_position, position, iterations):
    warm = model.decode_forward(
        tt_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=cache_position,
        position_index=position,
    )
    _assert_public_output_layout(warm)
    ttnn.synchronize_device(device)
    warm.deallocate(True)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    trace_output = model.decode_forward(
        tt_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=cache_position,
        position_index=position,
    )
    _assert_public_output_layout(trace_output)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    try:
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        for _ in range(iterations):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / iterations
        first = ttnn.to_torch(trace_output)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        second = ttnn.to_torch(trace_output)
    finally:
        ttnn.release_trace(device, trace_id)
    return elapsed_ms, first, second, trace_output


def _time_prefill(model, device, tt_hidden, key_cache, value_cache, iterations):
    for _ in range(2):
        output = model.prefill_forward(tt_hidden, key_cache=key_cache, value_cache=value_cache)
        ttnn.synchronize_device(device)
        output.deallocate(True)
    measurements = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = model.prefill_forward(tt_hidden, key_cache=key_cache, value_cache=value_cache)
        ttnn.synchronize_device(device)
        measurements.append((time.perf_counter() - start) * 1000.0)
        output.deallocate(True)
    return sorted(measurements)[len(measurements) // 2]


@pytest.mark.skipif(os.getenv("FALCON3_RUN_FINAL_PERF") != "1", reason="manual final performance gate")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64_000_000}], indirect=True)
@pytest.mark.timeout(1800)
def test_warmed_prefill_and_traced_decode_candidates(device):
    config = _config()
    batch = 32
    seq_len = EMITTED_PREFILL_SEQUENCE
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    hidden, decode_hidden = _recorded_layer20_inputs(batch)
    hf_cache = DynamicCache(config=config)
    expected_prefill = _hf_prefill(hf_layer, config, hidden, cache=hf_cache)
    expected_decode = _hf_decode(hf_layer, config, decode_hidden, hf_cache, seq_len)

    constructors = {
        "optimized_advisor_bfp8": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="bfp8_hifi2",
            decode_matmul_mode="shard_advisor",
            prefill_grid_x=8,
            prefill_in0_block_w=1,
        ),
        "optimized_selected_dram_all_bfp4_auto": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="all_bfp4_lofi",
            decode_matmul_mode="dram_sharded",
            prefill_grid_x=11,
            prefill_in0_block_w=8,
        ),
        "optimized_advisor_all_bfp4": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="all_bfp4_lofi",
            decode_matmul_mode="shard_advisor",
        ),
        "optimized_dram_all_bfp4_48c_control": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="all_bfp4_lofi",
            decode_matmul_mode="dram_sharded",
            dram_mlp_target_cores=48,
            prefill_grid_x=11,
            prefill_in0_block_w=8,
        ),
        "optimized_dram_all_bfp4_16c_control": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="all_bfp4_lofi",
            decode_matmul_mode="dram_sharded",
            dram_mlp_target_cores=16,
            prefill_grid_x=11,
            prefill_in0_block_w=8,
        ),
        "optimized_dram_all_bfp4_attention_hifi2": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="all_bfp4_attention_hifi2",
            decode_matmul_mode="dram_sharded",
            dram_mlp_target_cores=24,
            prefill_grid_x=11,
            prefill_in0_block_w=8,
        ),
        "optimized_dram_all_bfp4_mlp_hifi2": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="all_bfp4_mlp_hifi2",
            decode_matmul_mode="dram_sharded",
            dram_mlp_target_cores=24,
            prefill_grid_x=11,
            prefill_in0_block_w=8,
        ),
        "functional_bf16": lambda: FunctionalDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
        ),
    }
    results = {}
    for name, constructor in constructors.items():
        model = constructor()
        if "_dram_" in name:
            _assert_dram_decode_weight_contract(model)
        actual_prefill, key_cache, value_cache = _run_prefill(model, device, hidden)
        prefill_pcc = _assert_pcc(f"{name} prefill", expected_prefill, actual_prefill, SYNTHETIC_PCC)

        tt_prefill_hidden = _tt_input(hidden, device)
        prefill_ms = _time_prefill(model, device, tt_prefill_hidden, key_cache, value_cache, iterations=11)
        tt_prefill_hidden.deallocate(True)

        tt_decode_hidden = _tt_input(decode_hidden, device)
        cache_position = _cache_position(batch, seq_len, device)
        decode_ms, first, second, trace_output = _capture_and_time_decode(
            model,
            device,
            tt_decode_hidden,
            key_cache,
            value_cache,
            cache_position,
            seq_len,
            iterations=100,
        )
        decode_pcc = _assert_pcc(f"{name} traced decode", expected_decode, first.squeeze(0), SYNTHETIC_PCC)
        assert torch.equal(first, second), f"{name} trace replay is not deterministic"
        results[name] = {
            "prefill_pcc": prefill_pcc,
            "decode_pcc": decode_pcc,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
        }
        logger.info(
            f"PERF_RESULT batch={batch} {name}: " f"prefill_ms={prefill_ms:.6f} traced_decode_ms={decode_ms:.6f}"
        )

        trace_output.deallocate(True)
        tt_decode_hidden.deallocate(True)
        cache_position.deallocate(True)
        key_cache.deallocate(True)
        value_cache.deallocate(True)
        _release_model(model)

    selected = results["optimized_selected_dram_all_bfp4_auto"]
    for comparator in ("functional_bf16", "optimized_advisor_bfp8"):
        assert selected["prefill_ms"] < results[comparator]["prefill_ms"], results
        assert selected["decode_ms"] < results[comparator]["decode_ms"], results
    advisor = results["optimized_advisor_all_bfp4"]
    assert selected["decode_ms"] < advisor["decode_ms"], results
    assert selected["decode_ms"] < results["optimized_dram_all_bfp4_48c_control"]["decode_ms"], results
    assert selected["decode_ms"] < results["optimized_dram_all_bfp4_16c_control"]["decode_ms"], results
    assert selected["decode_ms"] < results["optimized_dram_all_bfp4_attention_hifi2"]["decode_ms"], results
    assert selected["decode_ms"] < results["optimized_dram_all_bfp4_mlp_hifi2"]["decode_ms"], results
    _write_result_artifact(
        "final_batch32.json",
        {"batch": batch, "sequence_length": seq_len, "weights": "real_layer_20", "results": results},
    )


@pytest.mark.skipif(os.getenv("FALCON3_RUN_FINAL_PERF") != "1", reason="manual final performance gate")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64_000_000}], indirect=True)
@pytest.mark.timeout(1800)
def test_batch1_traced_decode_candidates(device):
    """Measure primary single-user decode with identical real-weight cache fixtures."""
    config = _config()
    batch = 1
    seq_len = EMITTED_PREFILL_SEQUENCE
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    hidden, decode_hidden = _recorded_layer20_inputs(batch)
    hf_cache = DynamicCache(config=config)
    _hf_prefill(hf_layer, config, hidden, cache=hf_cache)
    hf_layer_cache = hf_cache.layers[IR_REPRESENTATIVE_LAYER]
    hf_key = hf_layer_cache.keys.clone()
    hf_value = hf_layer_cache.values.clone()
    expected_decode = _hf_decode(hf_layer, config, decode_hidden, hf_cache, seq_len)

    constructors = {
        "optimized_bf16": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="bf16_hifi4",
            decode_matmul_mode="shard_advisor",
        ),
        "optimized_dram_bfp8": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="bfp8_hifi2",
            decode_matmul_mode="dram_sharded",
            dram_mlp_target_cores=48,
        ),
        "optimized_advisor_all_bfp4": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="all_bfp4_lofi",
            decode_matmul_mode="shard_advisor",
        ),
        "optimized_selected_dram_all_bfp4_auto": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="all_bfp4_lofi",
            decode_matmul_mode="dram_sharded",
            prefill_grid_x=11,
            prefill_in0_block_w=8,
        ),
        "optimized_dram_all_bfp4_48c_control": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="all_bfp4_lofi",
            decode_matmul_mode="dram_sharded",
            dram_mlp_target_cores=48,
            prefill_grid_x=11,
            prefill_in0_block_w=8,
        ),
        "optimized_dram_all_bfp4_16c_control": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="all_bfp4_lofi",
            decode_matmul_mode="dram_sharded",
            dram_mlp_target_cores=16,
            prefill_grid_x=11,
            prefill_in0_block_w=8,
        ),
        "optimized_dram_all_bfp4_attention_hifi2": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="all_bfp4_attention_hifi2",
            decode_matmul_mode="dram_sharded",
            dram_mlp_target_cores=24,
            prefill_grid_x=11,
            prefill_in0_block_w=8,
        ),
        "optimized_dram_all_bfp4_mlp_hifi2": lambda: OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy="all_bfp4_mlp_hifi2",
            decode_matmul_mode="dram_sharded",
            dram_mlp_target_cores=24,
            prefill_grid_x=11,
            prefill_in0_block_w=8,
        ),
    }
    results = {}
    for name, constructor in constructors.items():
        model = constructor()
        if "_dram_" in name:
            _assert_dram_decode_weight_contract(model)
        key_cache, value_cache = _device_cache_from_hf(model, hf_key, hf_value, device)
        tt_decode_hidden = _tt_input(decode_hidden, device)
        cache_position = _cache_position(batch, seq_len, device)
        decode_ms, first, second, trace_output = _capture_and_time_decode(
            model,
            device,
            tt_decode_hidden,
            key_cache,
            value_cache,
            cache_position,
            seq_len,
            iterations=100,
        )
        decode_pcc = _assert_pcc(f"batch1 {name} traced decode", expected_decode, first.squeeze(0), REAL_WEIGHT_PCC)
        assert torch.equal(first, second), f"batch1 {name} trace replay is not deterministic"
        results[name] = {"decode_pcc": decode_pcc, "decode_ms": decode_ms}
        logger.info(f"PERF_RESULT batch=1 {name}: traced_decode_ms={decode_ms:.6f}")
        for tensor in (trace_output, tt_decode_hidden, cache_position, key_cache, value_cache):
            ttnn.deallocate(tensor)
        _release_model(model)

    selected = results["optimized_selected_dram_all_bfp4_auto"]["decode_ms"]
    assert selected < results["optimized_bf16"]["decode_ms"], results
    assert selected < results["optimized_dram_bfp8"]["decode_ms"], results
    assert selected < results["optimized_advisor_all_bfp4"]["decode_ms"], results
    assert selected < results["optimized_dram_all_bfp4_48c_control"]["decode_ms"], results
    assert selected < results["optimized_dram_all_bfp4_16c_control"]["decode_ms"], results
    assert selected < results["optimized_dram_all_bfp4_attention_hifi2"]["decode_ms"], results
    assert selected < results["optimized_dram_all_bfp4_mlp_hifi2"]["decode_ms"], results
    _write_result_artifact(
        "final_batch1.json",
        {"batch": batch, "sequence_length": seq_len, "weights": "real_layer_20", "results": results},
    )


@pytest.mark.skipif(
    os.getenv("FALCON3_RUN_RECORDED_PRECISION_FRONTIER") != "1", reason="manual recorded precision frontier"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64_000_000}], indirect=True)
@pytest.mark.timeout(1800)
def test_recorded_seq31_precision_frontier(device):
    """Select projection precision using real non-aligned activations, never random inputs."""
    config = _config()
    batch = 32
    seq_len = 31
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    hidden, decode_31, decode_32 = _recorded_layer20_seq31_inputs(batch)

    hf_cache = DynamicCache(config=config)
    expected_prefill = _hf_prefill(hf_layer, config, hidden, cache=hf_cache)
    prefill_layer_cache = hf_cache.layers[IR_REPRESENTATIVE_LAYER]
    expected_prefill_key = prefill_layer_cache.keys[:, :, :seq_len, :].clone()
    expected_prefill_value = prefill_layer_cache.values[:, :, :seq_len, :].clone()
    expected_query_31 = _hf_decode_query(hf_layer, config, decode_31, seq_len)
    expected_decode_31 = _hf_decode(hf_layer, config, decode_31, hf_cache, seq_len)
    expected_query_32 = _hf_decode_query(hf_layer, config, decode_32, seq_len + 1)
    expected_decode_32 = _hf_decode(hf_layer, config, decode_32, hf_cache, seq_len + 1)
    final_layer_cache = hf_cache.layers[IR_REPRESENTATIVE_LAYER]
    expected_final_key = final_layer_cache.keys[:, :, : seq_len + 2, :].clone()
    expected_final_value = final_layer_cache.values[:, :, : seq_len + 2, :].clone()

    candidates = {
        "mlp_bfp4_attention_hifi2": {
            "precision_policy": "mlp_bfp4_lofi",
            "prefill_grid_x": 11,
            "prefill_in0_block_w": 8,
        },
        "mlp_bfp4_attention_lofi": {
            "precision_policy": "mlp_bfp4_attention_lofi",
            "prefill_grid_x": 11,
            "prefill_in0_block_w": 8,
        },
        "attention_bfp4_mlp_bfp8": {
            "precision_policy": "attention_bfp4_lofi",
            "prefill_grid_x": 8,
            "prefill_in0_block_w": 1,
        },
        "all_projections_bfp4_lofi": {
            "precision_policy": "all_bfp4_lofi",
            "prefill_grid_x": 11,
            "prefill_in0_block_w": 8,
        },
        "all_bfp4_attention_hifi2": {
            "precision_policy": "all_bfp4_attention_hifi2",
            "prefill_grid_x": 11,
            "prefill_in0_block_w": 8,
        },
    }
    results = {}
    for name, identity in candidates.items():
        model = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            decode_matmul_mode="shard_advisor",
            **identity,
        )
        actual_prefill, key_cache, value_cache = _run_prefill(model, device, hidden)
        _, prefill_pcc = comp_pcc(expected_prefill.float(), actual_prefill.float(), pcc=0.0)
        _, prefill_key_pcc = comp_pcc(
            expected_prefill_key.float(), ttnn.to_torch(key_cache)[:, :, :seq_len, :].float(), pcc=0.0
        )
        _, prefill_value_pcc = comp_pcc(
            expected_prefill_value.float(), ttnn.to_torch(value_cache)[:, :, :seq_len, :].float(), pcc=0.0
        )
        actual_query_31 = _run_decode_query(model, device, decode_31, seq_len)
        _, query_31_pcc = comp_pcc(expected_query_31.float(), actual_query_31.float(), pcc=0.0)

        tt_prefill_hidden = _tt_input(hidden, device)
        prefill_ms = _time_prefill(model, device, tt_prefill_hidden, key_cache, value_cache, iterations=3)
        ttnn.deallocate(tt_prefill_hidden)
        tt_decode_hidden = _tt_input(decode_31, device)
        cache_position = _cache_position(batch, seq_len, device)
        decode_ms, first, repeated, trace_output = _capture_and_time_decode(
            model,
            device,
            tt_decode_hidden,
            key_cache,
            value_cache,
            cache_position,
            seq_len,
            iterations=20,
        )
        assert torch.equal(first, repeated), f"{name} trace replay is not deterministic"
        _, decode_31_pcc = comp_pcc(expected_decode_31.float(), first.squeeze(0).float(), pcc=0.0)

        actual_query_32 = _run_decode_query(model, device, decode_32, seq_len + 1)
        _, query_32_pcc = comp_pcc(expected_query_32.float(), actual_query_32.float(), pcc=0.0)
        actual_decode_32 = _run_decode(model, device, decode_32, key_cache, value_cache, seq_len + 1)
        _, decode_32_pcc = comp_pcc(expected_decode_32.float(), actual_decode_32.float(), pcc=0.0)
        _, final_key_pcc = comp_pcc(
            expected_final_key.float(), ttnn.to_torch(key_cache)[:, :, : seq_len + 2, :].float(), pcc=0.0
        )
        _, final_value_pcc = comp_pcc(
            expected_final_value.float(), ttnn.to_torch(value_cache)[:, :, : seq_len + 2, :].float(), pcc=0.0
        )
        results[name] = {
            "identity": {**identity, "decode_matmul_mode": "shard_advisor"},
            "prefill_pcc": prefill_pcc,
            "prefill_key_cache_pcc": prefill_key_pcc,
            "prefill_value_cache_pcc": prefill_value_pcc,
            "decode_31_query_pcc": query_31_pcc,
            "decode_31_pcc": decode_31_pcc,
            "decode_32_query_pcc": query_32_pcc,
            "decode_32_pcc": decode_32_pcc,
            "final_key_cache_pcc": final_key_pcc,
            "final_value_cache_pcc": final_value_pcc,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
        }
        logger.info(
            f"RECORDED_SEQ31_RESULT {name}: prefill={prefill_pcc:.8f} "
            f"decode31={decode_31_pcc:.8f} decode32={decode_32_pcc:.8f} "
            f"query31={query_31_pcc:.8f} query32={query_32_pcc:.8f} "
            f"final_k={final_key_pcc:.8f} final_v={final_value_pcc:.8f} "
            f"prefill_ms={prefill_ms:.6f} decode_ms={decode_ms:.6f}"
        )
        for tensor in (trace_output, tt_decode_hidden, cache_position, key_cache, value_cache):
            ttnn.deallocate(tensor)
        _release_model(model)

    _write_result_artifact(
        "recorded_seq31_precision_frontier.json",
        {
            "activation_fixture": str(ACTIVATION_FIXTURE),
            "batch": batch,
            "layer": IR_REPRESENTATIVE_LAYER,
            "sequence_length": seq_len,
            "weights": "real_layer_20",
            "inputs": "HF embedding plus layers 0..19 for a genuine 31-token prompt and two following tokens",
            "results": results,
        },
    )


@pytest.mark.skipif(os.getenv("FALCON3_RUN_CANDIDATE_SWEEP") != "1", reason="manual optimization sweep")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64_000_000}], indirect=True)
@pytest.mark.timeout(1800)
def test_decode_candidate_sweep(device):
    config = _config()
    batch = 32
    seq_len = EMITTED_PREFILL_SEQUENCE
    use_real_weights = os.getenv("FALCON3_REAL_WEIGHTS") == "1"
    state_dict = (
        _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
        if use_real_weights
        else _synthetic_state_dict(config, IR_REPRESENTATIVE_LAYER)
    )
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    input_source = os.getenv("FALCON3_INPUT_SOURCE", "recorded")
    if input_source == "recorded":
        hidden, decode_hidden = _recorded_layer20_inputs(batch)
    elif input_source == "random_sensitivity":
        generator = torch.Generator().manual_seed(314)
        hidden = torch.randn((batch, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
        decode_hidden = torch.randn((batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported FALCON3_INPUT_SOURCE={input_source!r}")
    hf_cache = DynamicCache(config=config)
    expected_prefill = _hf_prefill(hf_layer, config, hidden, cache=hf_cache)
    hf_layer_cache = hf_cache.layers[IR_REPRESENTATIVE_LAYER]
    expected_key_cache = hf_layer_cache.keys[:, :, :seq_len, :].clone()
    expected_value_cache = hf_layer_cache.values[:, :, :seq_len, :].clone()
    expected_decode_query = _hf_decode_query(hf_layer, config, decode_hidden, seq_len)
    expected_decode = _hf_decode(hf_layer, config, decode_hidden, hf_cache, seq_len)

    candidates = {
        "advisor_bfp8": {},
        "advisor_bfp8_lofi": {"precision_policy": "bfp8_lofi"},
        "advisor_attention_bfp8_lofi": {"precision_policy": "attention_bfp8_lofi"},
        "advisor_mlp_bfp8_lofi": {"precision_policy": "mlp_bfp8_lofi"},
        "advisor_attention_bfp4": {"precision_policy": "attention_bfp4_lofi"},
        "advisor_mlp_bfp4": {"precision_policy": "mlp_bfp4_lofi"},
        "selected_attention_lofi": {"precision_policy": "mlp_bfp4_attention_lofi"},
        "selected_mlp_hifi2": {"precision_policy": "mlp_bfp4_hifi2"},
        "advisor_all_bfp4": {"precision_policy": "all_bfp4_lofi"},
        "advisor_all_bfp4_grid8": {
            "precision_policy": "all_bfp4_lofi",
            "prefill_grid_x": 8,
            "prefill_in0_block_w": 8,
        },
        "advisor_explicit_mask": {"use_explicit_decode_mask": True},
        "advisor_wide_blocks": {
            "precision_policy": "mlp_bfp4_lofi",
            "advisor_mlp_geometry": "wide_blocks",
        },
        "advisor_wider_blocks": {
            "precision_policy": "mlp_bfp4_lofi",
            "advisor_mlp_geometry": "wider_blocks",
        },
        "advisor_packed_mlp": {
            "precision_policy": "mlp_bfp4_lofi",
            "use_packed_mlp": True,
            "prefill_in0_block_w": 1,
        },
        "advisor_residual_legacy": {
            "precision_policy": "mlp_bfp4_lofi",
            "advisor_residual_mode": "legacy_32core",
        },
        "advisor_residual_report": {
            "precision_policy": "mlp_bfp4_lofi",
            "advisor_residual_mode": "report",
        },
        "advisor_report_sharded_inputs": {
            "precision_policy": "mlp_bfp4_lofi",
            "advisor_residual_mode": "report",
            "advisor_matmul_input_mode": "report_sharded",
        },
        "advisor_large_prefill_grid": {"precision_policy": "mlp_bfp4_lofi", "prefill_grid_x": 11},
        "prefill_grid8": {
            "precision_policy": "mlp_bfp4_lofi",
            "prefill_grid_x": 8,
            "prefill_in0_block_w": 8,
        },
        "prefill_block1": {"precision_policy": "mlp_bfp4_lofi", "prefill_in0_block_w": 1},
        "prefill_block2": {"precision_policy": "mlp_bfp4_lofi", "prefill_in0_block_w": 2},
        "prefill_block4": {"precision_policy": "mlp_bfp4_lofi", "prefill_in0_block_w": 4},
        "prefill_block8": {"precision_policy": "mlp_bfp4_lofi", "prefill_in0_block_w": 8},
        "prefill_block16": {"precision_policy": "mlp_bfp4_lofi", "prefill_in0_block_w": 16},
        "prefill_block16_grid8": {
            "precision_policy": "mlp_bfp4_lofi",
            "prefill_grid_x": 8,
            "prefill_in0_block_w": 16,
        },
        "dram_packed_bfp8": {"decode_matmul_mode": "dram_sharded", "use_packed_mlp": True},
        "dram_mlp_bfp4_24c": {
            "decode_matmul_mode": "dram_sharded",
            "precision_policy": "mlp_bfp4_lofi",
            "dram_mlp_target_cores": 24,
        },
        "dram_mlp_bfp4_48c": {
            "decode_matmul_mode": "dram_sharded",
            "precision_policy": "mlp_bfp4_lofi",
            "dram_mlp_target_cores": 48,
        },
        "dram_all_bfp4_24c": {
            "decode_matmul_mode": "dram_sharded",
            "precision_policy": "all_bfp4_lofi",
            "dram_mlp_target_cores": 24,
        },
        "dram_all_bfp4_12c": {
            "decode_matmul_mode": "dram_sharded",
            "precision_policy": "all_bfp4_lofi",
            "dram_mlp_target_cores": 12,
        },
        "dram_all_bfp4_16c": {
            "decode_matmul_mode": "dram_sharded",
            "precision_policy": "all_bfp4_lofi",
            "dram_mlp_target_cores": 16,
        },
        "dram_all_bfp4_6c": {
            "decode_matmul_mode": "dram_sharded",
            "precision_policy": "all_bfp4_lofi",
            "dram_mlp_target_cores": 6,
        },
        "dram_all_bfp4_24c_attention_hifi2": {
            "decode_matmul_mode": "dram_sharded",
            "precision_policy": "all_bfp4_attention_hifi2",
            "dram_mlp_target_cores": 24,
        },
        "dram_all_bfp4_24c_mlp_hifi2": {
            "decode_matmul_mode": "dram_sharded",
            "precision_policy": "all_bfp4_mlp_hifi2",
            "dram_mlp_target_cores": 24,
        },
        "dram_all_bfp4_48c": {
            "decode_matmul_mode": "dram_sharded",
            "precision_policy": "all_bfp4_lofi",
            "dram_mlp_target_cores": 48,
        },
        "dram_all_bfp4_48c_unaligned_down": {
            "decode_matmul_mode": "dram_sharded",
            "precision_policy": "all_bfp4_lofi",
            "dram_mlp_target_cores": 48,
            "align_dram_mlp_down_input": False,
        },
        "dram_all_bfp4_48c_packed": {
            "decode_matmul_mode": "dram_sharded",
            "precision_policy": "all_bfp4_lofi",
            "dram_mlp_target_cores": 48,
            "use_packed_mlp": True,
            "prefill_in0_block_w": 1,
        },
    }
    selected_candidates = os.getenv("FALCON3_CANDIDATES")
    if selected_candidates:
        selected_candidates = {name.strip() for name in selected_candidates.split(",") if name.strip()}
        candidates = {name: value for name, value in candidates.items() if name in selected_candidates}
        assert candidates, f"FALCON3_CANDIDATES selected no known candidate: {selected_candidates}"
    results = {}
    for name, overrides in candidates.items():
        options = dict(overrides)
        precision_policy = options.pop("precision_policy", "bfp8_hifi2")
        options.setdefault("decode_matmul_mode", "shard_advisor")
        bfp4_mlp_policy = precision_policy in (
            "mlp_bfp4_lofi",
            "mlp_bfp4_attention_lofi",
            "mlp_bfp4_hifi2",
            "all_bfp4_lofi",
            "all_bfp4_attention_hifi2",
            "all_bfp4_mlp_hifi2",
        )
        options.setdefault("prefill_in0_block_w", 8 if bfp4_mlp_policy else 1)
        options.setdefault("prefill_grid_x", 11 if bfp4_mlp_policy else 8)
        model = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            precision_policy=precision_policy,
            **options,
        )
        actual_prefill, key_cache, value_cache = _run_prefill(model, device, hidden)
        _, prefill_pcc = comp_pcc(expected_prefill.float(), actual_prefill.float(), pcc=0.0)
        actual_key_cache = ttnn.to_torch(key_cache)[:, :, :seq_len, :]
        actual_value_cache = ttnn.to_torch(value_cache)[:, :, :seq_len, :]
        _, key_cache_pcc = comp_pcc(expected_key_cache.float(), actual_key_cache.float(), pcc=0.0)
        _, value_cache_pcc = comp_pcc(expected_value_cache.float(), actual_value_cache.float(), pcc=0.0)
        actual_decode_query = _run_decode_query(model, device, decode_hidden, seq_len)
        _, decode_query_pcc = comp_pcc(expected_decode_query.float(), actual_decode_query.float(), pcc=0.0)

        tt_prefill_hidden = _tt_input(hidden, device)
        prefill_ms = _time_prefill(model, device, tt_prefill_hidden, key_cache, value_cache, iterations=3)
        ttnn.deallocate(tt_prefill_hidden)

        tt_decode_hidden = _tt_input(decode_hidden, device)
        cache_position = _cache_position(batch, seq_len, device)
        candidate_decode_iterations = int(os.getenv("FALCON3_CANDIDATE_DECODE_ITERATIONS", "20"))
        decode_ms, first, second, trace_output = _capture_and_time_decode(
            model,
            device,
            tt_decode_hidden,
            key_cache,
            value_cache,
            cache_position,
            seq_len,
            iterations=candidate_decode_iterations,
        )
        _, decode_pcc = comp_pcc(expected_decode.float(), first.squeeze(0).float(), pcc=0.0)
        assert torch.equal(first, second), f"{name} trace replay is not deterministic"
        logger.info(
            f"CANDIDATE_RESULT {name}: prefill_pcc={prefill_pcc:.8f} "
            f"decode_pcc={decode_pcc:.8f} query_pcc={decode_query_pcc:.8f} "
            f"key_cache_pcc={key_cache_pcc:.8f} value_cache_pcc={value_cache_pcc:.8f} "
            f"prefill_ms={prefill_ms:.6f} traced_decode_ms={decode_ms:.6f}"
        )
        results[name] = {
            "prefill_pcc": prefill_pcc,
            "decode_pcc": decode_pcc,
            "decode_query_pcc": decode_query_pcc,
            "key_cache_pcc": key_cache_pcc,
            "value_cache_pcc": value_cache_pcc,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "precision_policy": precision_policy,
            "options": options,
        }

        trace_output.deallocate(True)
        tt_decode_hidden.deallocate(True)
        cache_position.deallocate(True)
        key_cache.deallocate(True)
        value_cache.deallocate(True)
        _release_model(model)
    _write_result_artifact(
        "candidate_sweep.json",
        {
            "batch": batch,
            "sequence_length": seq_len,
            "weights": "real_layer_20" if use_real_weights else "synthetic",
            "inputs": input_source,
            "results": results,
        },
    )


@pytest.mark.skipif(os.getenv("FALCON3_RUN_DRAM_BFP4_DIAGNOSTIC") != "1", reason="manual AutoFix diagnostic")
@pytest.mark.timeout(1800)
def test_dram_bfp4_weight_materialization(device):
    """Localize DRAM-BFP4 failure before changing matmul/runtime code."""
    config = _config()
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    prefix = f"model.layers.{IR_REPRESENTATIVE_LAYER}."
    sources = {
        "gate_weight": state_dict[prefix + "mlp.gate_proj.weight"].T.contiguous(),
        "up_weight": state_dict[prefix + "mlp.up_proj.weight"].T.contiguous(),
        "down_weight": state_dict[prefix + "mlp.down_proj.weight"].T.contiguous(),
    }
    constructors = {
        "advisor_bfp4": {"precision_policy": "mlp_bfp4_lofi", "decode_matmul_mode": "shard_advisor"},
        "dram_bfp4": {"precision_policy": "mlp_bfp4_lofi", "decode_matmul_mode": "dram_sharded"},
        "dram_bfp8": {
            "precision_policy": "bfp8_hifi2",
            "decode_matmul_mode": "dram_sharded",
            "prefill_grid_x": 8,
            "prefill_in0_block_w": 1,
        },
    }
    results = {}
    for candidate, options in constructors.items():
        model = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=32,
            **options,
        )
        results[candidate] = {}
        for name, expected in sources.items():
            decode_name = name.replace("_weight", "_decode_weight")
            device_weight = getattr(model, decode_name, None)
            if device_weight is None:
                device_weight = getattr(model, name)
            actual = ttnn.to_torch(device_weight).reshape(expected.shape)
            _, pcc = comp_pcc(expected.float(), actual.float(), pcc=0.0)
            results[candidate][name] = {
                "pcc": pcc,
                "expected_abs_max": float(expected.abs().max()),
                "actual_abs_max": float(actual.abs().max()),
                "actual_mean": float(actual.float().mean()),
                "actual_std": float(actual.float().std()),
            }
            logger.info(f"MATERIALIZATION_RESULT {candidate} {name}: PCC={pcc:.8f}")
        _release_model(model)
    _write_result_artifact(
        "weight_materialization.json",
        {"weights": "real_layer_20", "results": results},
    )


@pytest.mark.skipif(os.getenv("FALCON3_RUN_DRAM_BFP4_DIAGNOSTIC") != "1", reason="manual AutoFix diagnostic")
@pytest.mark.timeout(1800)
def test_dram_bfp4_matmul_localization(device):
    """Compare each DRAM-sharded MLP matmul against the same HF boundary."""
    config = _config()
    batch = 32
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    _, decode_hidden = _recorded_layer20_inputs(batch)
    normed_ref = hf_layer.post_attention_layernorm(decode_hidden)
    gate_ref = hf_layer.mlp.gate_proj(normed_ref)
    up_ref = hf_layer.mlp.up_proj(normed_ref)
    gated_ref = torch.nn.functional.silu(gate_ref) * up_ref
    down_ref = hf_layer.mlp.down_proj(gated_ref)
    constructors = {
        "advisor_bfp4": {"precision_policy": "mlp_bfp4_lofi", "decode_matmul_mode": "shard_advisor"},
        "dram_bfp4_24c": {
            "precision_policy": "mlp_bfp4_lofi",
            "decode_matmul_mode": "dram_sharded",
            "dram_mlp_target_cores": 24,
        },
        "dram_bfp4_48c": {
            "precision_policy": "mlp_bfp4_lofi",
            "decode_matmul_mode": "dram_sharded",
            "dram_mlp_target_cores": 48,
        },
        "dram_bfp8": {
            "precision_policy": "bfp8_hifi2",
            "decode_matmul_mode": "dram_sharded",
            "prefill_grid_x": 8,
            "prefill_in0_block_w": 1,
        },
    }
    results = {}
    for candidate, options in constructors.items():
        model = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            **options,
        )
        tt_hidden = _tt_input(decode_hidden, device)
        residual = ttnn.reshape(tt_hidden, (1, 1, batch, config.hidden_size))
        residual = ttnn.to_memory_config(residual, model.decode_input_memory_config)
        normed = model._decode_norm(residual, model.post_attention_norm_weight)
        mlp_input = model._move_owned(normed, model.gate_input_memory_config)
        gate_weight = model.gate_decode_weight if model.gate_decode_weight is not None else model.gate_weight
        up_weight = model.up_decode_weight if model.up_decode_weight is not None else model.up_weight
        gate = ttnn.matmul(
            mlp_input,
            gate_weight,
            dtype=ttnn.bfloat16,
            program_config=model.gate_decode_program_config,
            compute_kernel_config=model.mlp_compute_config,
            memory_config=model.mlp_gate_output_memory_config,
        )
        up = ttnn.matmul(
            mlp_input,
            up_weight,
            dtype=ttnn.bfloat16,
            program_config=model.gate_decode_program_config,
            compute_kernel_config=model.mlp_compute_config,
            memory_config=model.mlp_gate_output_memory_config,
        )
        actual_gate = ttnn.to_torch(gate).reshape(gate_ref.shape)
        actual_up = ttnn.to_torch(up).reshape(up_ref.shape)
        _, gate_pcc = comp_pcc(gate_ref.float(), actual_gate.float(), pcc=0.0)
        _, up_pcc = comp_pcc(up_ref.float(), actual_up.float(), pcc=0.0)
        gated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=model.mlp_gate_output_memory_config,
        )
        actual_gated = ttnn.to_torch(gated).reshape(gated_ref.shape)
        _, gated_pcc = comp_pcc(gated_ref.float(), actual_gated.float(), pcc=0.0)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        ttnn.deallocate(mlp_input)
        ttnn.deallocate(residual)
        ttnn.deallocate(tt_hidden)

        down_input = model._move_owned(gated, model.down_input_memory_config)
        down = ttnn.matmul(
            down_input,
            model.down_decode_weight if model.down_decode_weight is not None else model.down_weight,
            dtype=ttnn.bfloat16,
            program_config=model.down_decode_program_config,
            compute_kernel_config=model.mlp_compute_config,
            memory_config=model.mlp_down_output_memory_config,
        )
        actual_down = ttnn.to_torch(down).reshape(down_ref.shape)
        _, down_pcc = comp_pcc(down_ref.float(), actual_down.float(), pcc=0.0)
        results[candidate] = {
            "gate_pcc": gate_pcc,
            "up_pcc": up_pcc,
            "gated_pcc": gated_pcc,
            "down_pcc": down_pcc,
        }
        logger.info(
            f"MATMUL_LOCALIZATION {candidate}: gate={gate_pcc:.8f} up={up_pcc:.8f} "
            f"gated={gated_pcc:.8f} down={down_pcc:.8f}"
        )
        ttnn.deallocate(down)
        ttnn.deallocate(down_input)
        _release_model(model)
    _write_result_artifact(
        "matmul_localization.json",
        {"weights": "real_layer_20", "inputs": "recorded_layer20_activations", "results": results},
    )


@pytest.mark.skipif(os.getenv("FALCON3_RUN_DRAM_BFP4_DIAGNOSTIC") != "1", reason="manual AutoFix diagnostic")
@pytest.mark.timeout(1800)
def test_dram_bfp4_stage_localization(device):
    """Locate full-decode divergence at the attention/MLP stage boundary."""
    config = _config()
    batch = 32
    seq_len = EMITTED_PREFILL_SEQUENCE
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    hidden, decode_hidden = _recorded_layer20_inputs(batch)
    hf_cache = DynamicCache(config=config)
    _hf_prefill(hf_layer, config, hidden, cache=hf_cache)
    hf_layer_cache = hf_cache.layers[IR_REPRESENTATIVE_LAYER]
    hf_key = hf_layer_cache.keys.clone()
    hf_value = hf_layer_cache.values.clone()
    captured = {}

    def capture_attention_residual(_module, args):
        captured["attention_residual"] = args[0].detach().clone()

    handle = hf_layer.post_attention_layernorm.register_forward_pre_hook(capture_attention_residual)
    expected = _hf_decode(hf_layer, config, decode_hidden, hf_cache, seq_len)
    handle.remove()
    expected_attention = captured["attention_residual"]

    constructors = {
        "advisor_bfp4": {"precision_policy": "mlp_bfp4_lofi", "decode_matmul_mode": "shard_advisor"},
        "dram_bfp4_24c": {
            "precision_policy": "mlp_bfp4_lofi",
            "decode_matmul_mode": "dram_sharded",
            "dram_mlp_target_cores": 24,
        },
        "dram_bfp4_48c": {
            "precision_policy": "mlp_bfp4_lofi",
            "decode_matmul_mode": "dram_sharded",
            "dram_mlp_target_cores": 48,
        },
        "dram_bfp8": {"precision_policy": "bfp8_hifi2", "decode_matmul_mode": "dram_sharded"},
    }
    results = {}
    for candidate, options in constructors.items():
        model = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            **options,
        )
        key_cache, value_cache = _device_cache_from_hf(model, hf_key, hf_value, device)
        tt_hidden = _tt_input(decode_hidden, device)
        cache_position = _cache_position(batch, seq_len, device)
        residual = ttnn.reshape(tt_hidden, (1, 1, batch, config.hidden_size))
        residual = ttnn.to_memory_config(residual, model.decode_input_memory_config)
        attention_residual = model._decode_attention(
            residual,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=cache_position,
            position_index=seq_len,
        )
        actual_attention = ttnn.to_torch(attention_residual).reshape(expected_attention.shape)
        _, attention_pcc = comp_pcc(expected_attention.float(), actual_attention.float(), pcc=0.0)
        output = model._decode_mlp(attention_residual)
        actual_output = ttnn.to_torch(output).reshape(expected.shape)
        _, output_pcc = comp_pcc(expected.float(), actual_output.float(), pcc=0.0)
        results[candidate] = {"attention_residual_pcc": attention_pcc, "full_output_pcc": output_pcc}
        logger.info(f"STAGE_LOCALIZATION {candidate}: attention={attention_pcc:.8f} output={output_pcc:.8f}")
        for tensor in (output, tt_hidden, cache_position, key_cache, value_cache):
            ttnn.deallocate(tensor)
        _release_model(model)
    _write_result_artifact(
        "stage_localization.json",
        {"weights": "real_layer_20", "inputs": "recorded_layer20_activations", "results": results},
    )


@pytest.mark.skipif(os.getenv("FALCON3_RUN_DRAM_BFP4_DIAGNOSTIC") != "1", reason="manual AutoFix diagnostic")
@pytest.mark.timeout(1800)
def test_dram_bfp4_prefill_stage_localization(device):
    """Locate prefill divergence and verify whether it corrupts KV caches."""
    config = _config()
    batch = 32
    seq_len = EMITTED_PREFILL_SEQUENCE
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    hf_layer = _hf_layer(config, state_dict, IR_REPRESENTATIVE_LAYER)
    hidden, _ = _recorded_layer20_inputs(batch)
    captured = {}

    def capture_attention_residual(_module, args):
        captured["attention_residual"] = args[0].detach().clone()

    hf_cache = DynamicCache(config=config)
    handle = hf_layer.post_attention_layernorm.register_forward_pre_hook(capture_attention_residual)
    expected = _hf_prefill(hf_layer, config, hidden, cache=hf_cache)
    handle.remove()
    expected_attention = captured["attention_residual"]
    hf_layer_cache = hf_cache.layers[IR_REPRESENTATIVE_LAYER]

    constructors = {
        "advisor_bfp4": {"precision_policy": "mlp_bfp4_lofi", "decode_matmul_mode": "shard_advisor"},
        "dram_bfp4_24c": {
            "precision_policy": "mlp_bfp4_lofi",
            "decode_matmul_mode": "dram_sharded",
            "dram_mlp_target_cores": 24,
        },
        "dram_bfp4_48c": {
            "precision_policy": "mlp_bfp4_lofi",
            "decode_matmul_mode": "dram_sharded",
            "dram_mlp_target_cores": 48,
        },
        "dram_bfp8": {
            "precision_policy": "bfp8_hifi2",
            "decode_matmul_mode": "dram_sharded",
            "prefill_grid_x": 8,
            "prefill_in0_block_w": 1,
        },
    }
    results = {}
    for candidate, options in constructors.items():
        model = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=config,
            layer_idx=IR_REPRESENTATIVE_LAYER,
            mesh_device=device,
            batch=batch,
            **options,
        )
        if candidate.startswith("dram_"):
            _assert_dram_decode_weight_contract(model)
        key_cache, value_cache = model.allocate_kv_cache()
        tt_hidden = _tt_input(hidden, device)
        residual = ttnn.reshape(tt_hidden, (1, 1, batch * seq_len, config.hidden_size))
        attention_residual = model._prefill_attention(
            residual,
            batch=batch,
            seq_len=seq_len,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        actual_attention = ttnn.to_torch(attention_residual).reshape(expected_attention.shape)
        _, attention_pcc = comp_pcc(expected_attention.float(), actual_attention.float(), pcc=0.0)
        output = model._prefill_mlp(attention_residual)
        actual_output = ttnn.to_torch(output).reshape(expected.shape)
        _, output_pcc = comp_pcc(expected.float(), actual_output.float(), pcc=0.0)
        actual_key = ttnn.to_torch(key_cache)[:, :, :seq_len, :]
        actual_value = ttnn.to_torch(value_cache)[:, :, :seq_len, :]
        _, key_pcc = comp_pcc(hf_layer_cache.keys.float(), actual_key.float(), pcc=0.0)
        _, value_pcc = comp_pcc(hf_layer_cache.values.float(), actual_value.float(), pcc=0.0)
        results[candidate] = {
            "attention_residual_pcc": attention_pcc,
            "full_output_pcc": output_pcc,
            "key_cache_pcc": key_pcc,
            "value_cache_pcc": value_pcc,
        }
        logger.info(
            f"PREFILL_LOCALIZATION {candidate}: attention={attention_pcc:.8f} output={output_pcc:.8f} "
            f"key={key_pcc:.8f} value={value_pcc:.8f}"
        )
        for tensor in (output, tt_hidden, key_cache, value_cache):
            ttnn.deallocate(tensor)
        _release_model(model)
    _write_result_artifact(
        "prefill_stage_localization.json",
        {"weights": "real_layer_20", "inputs": "recorded_layer20_activations", "results": results},
    )


@pytest.mark.skipif(os.getenv("FALCON3_RUN_PROFILE") != "1", reason="manual Tracy profile")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64_000_000}], indirect=True)
@pytest.mark.timeout(1800)
def test_profile_selected_decoder(device):
    """Capture one representative layer with separate prefill/decode signposts."""
    from tracy import signpost

    config = _config()
    batch = 32
    seq_len = EMITTED_PREFILL_SEQUENCE
    state_dict = _real_layer_state_dict(IR_REPRESENTATIVE_LAYER)
    model = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=IR_REPRESENTATIVE_LAYER,
        mesh_device=device,
        batch=batch,
    )
    hidden, decode_hidden = _recorded_layer20_inputs(batch)
    _, key_cache, value_cache = _run_prefill(model, device, hidden)

    tt_prefill_hidden = _tt_input(hidden, device)
    warm_prefill = model.prefill_forward(tt_prefill_hidden, key_cache=key_cache, value_cache=value_cache)
    ttnn.synchronize_device(device)
    ttnn.deallocate(warm_prefill)
    # Keep the device-profiler ring bounded so repeated measured regions retain
    # every marker. Reads append to the same Tracy report and clear the ring.
    ttnn.ReadDeviceProfiler(device)
    prefill_iterations = 2
    signpost(header="PERF_PREFILL")
    prefill_start = time.perf_counter()
    for _ in range(prefill_iterations):
        profile_prefill = model.prefill_forward(tt_prefill_hidden, key_cache=key_cache, value_cache=value_cache)
        ttnn.synchronize_device(device)
        ttnn.deallocate(profile_prefill)
    prefill_wall_total_ms = (time.perf_counter() - prefill_start) * 1000.0
    signpost(header="PERF_PREFILL_END")
    ttnn.ReadDeviceProfiler(device)

    tt_decode_hidden = _tt_input(decode_hidden, device)
    cache_position = _cache_position(batch, seq_len, device)
    warm_decode = model.decode_forward(
        tt_decode_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=cache_position,
        position_index=seq_len,
    )
    ttnn.synchronize_device(device)
    ttnn.deallocate(warm_decode)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    trace_output = model.decode_forward(
        tt_decode_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=cache_position,
        position_index=seq_len,
    )
    _assert_public_output_layout(trace_output)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    decode_iterations = 3
    try:
        signpost(header="PERF_DECODE")
        decode_start = time.perf_counter()
        for _ in range(decode_iterations):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        decode_wall_total_ms = (time.perf_counter() - decode_start) * 1000.0
        signpost(header="PERF_DECODE_END")
        ttnn.ReadDeviceProfiler(device)
    finally:
        ttnn.release_trace(device, trace_id)

    for tensor in (
        trace_output,
        tt_prefill_hidden,
        tt_decode_hidden,
        cache_position,
        key_cache,
        value_cache,
    ):
        ttnn.deallocate(tensor)
    _release_model(model)
    _write_result_artifact(
        "profile_wall.json",
        {
            "batch": batch,
            "sequence_length": seq_len,
            "weights": "real_layer_20",
            "prefill_iterations": prefill_iterations,
            "prefill_wall_total_ms": prefill_wall_total_ms,
            "prefill_wall_ms": prefill_wall_total_ms / prefill_iterations,
            "decode_iterations": decode_iterations,
            "decode_wall_total_ms": decode_wall_total_ms,
            "decode_wall_ms": decode_wall_total_ms / decode_iterations,
        },
    )
