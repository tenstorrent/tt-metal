# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import inspect
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

import models.autoports.qwen_qwen3_32b.tests.test_functional_decoder as functional_test
import ttnn
from models.autoports.qwen_qwen3_32b.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    REPRESENTATIVE_LAYER,
    FunctionalDecoder,
)
from models.autoports.qwen_qwen3_32b.tt.fused_decoder import FusedDecoder

RUN_PERF_ENV = "QWEN3_32B_RUN_FUSED_PERF"
RESULTS_DIR_ENV = "QWEN3_32B_FUSED_RESULTS_DIR"
CONTEXT_PROBE_ENV = "QWEN3_32B_FUSED_CONTEXT_PROBE_LEN"


@contextmanager
def _functional_helpers_for_layer(layer_idx: int):
    previous = functional_test.REPRESENTATIVE_LAYER
    functional_test.REPRESENTATIVE_LAYER = layer_idx
    try:
        yield
    finally:
        functional_test.REPRESENTATIVE_LAYER = previous


def _decoder(decoder_cls, state, config, mesh_device, *, layer_idx=REPRESENTATIVE_LAYER):
    return decoder_cls.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )


def _assert_preserves_functional_pcc(reference, functional, fused, threshold: float, label: str):
    functional_passed, functional_pcc = functional_test.comp_pcc(reference.float(), functional.float(), pcc=threshold)
    fused_passed, fused_pcc = functional_test.comp_pcc(reference.float(), fused.float(), pcc=threshold)
    _, functional_to_fused_pcc = functional_test.comp_pcc(functional.float(), fused.float(), pcc=0.9999)
    print(f"{label}: functional={functional_pcc}, fused={fused_pcc}, " f"functional_to_fused={functional_to_fused_pcc}")
    if functional_passed:
        assert fused_passed, f"{label}: fused PCC {fused_pcc} is below the functional bar {threshold}"
    else:
        assert (
            fused_pcc >= functional_pcc - 1.0e-6
        ), f"{label}: fused PCC {fused_pcc} regresses the below-bar functional baseline {functional_pcc}"
    assert functional_to_fused_pcc >= 0.9999
    return fused_pcc


def _allocated_dram_bytes(mesh_device) -> int:
    ttnn.synchronize_device(mesh_device)
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return int(view.num_banks * view.total_bytes_allocated_per_bank)


def test_fused_runtime_is_distinct_and_host_free():
    assert FusedDecoder.prefill_forward is not FunctionalDecoder.prefill_forward
    assert FusedDecoder.decode_forward is not FunctionalDecoder.decode_forward
    assert FusedDecoder._mlp_forward is not FunctionalDecoder._mlp_forward

    methods = (
        FusedDecoder._get_prefill_rotary_views,
        FusedDecoder._get_decode_position_views,
        FusedDecoder._mlp_forward,
        FusedDecoder._update_cache,
        FusedDecoder._create_decode_heads,
        FusedDecoder._normalize_and_rotate_decode_qk,
        FusedDecoder._decode_attention,
        FusedDecoder._concatenate_decode_heads,
        FusedDecoder.prefill_forward,
        FusedDecoder.decode_forward,
    )
    forbidden = (
        "torch",
        "from_torch",
        "to_torch",
        "numpy",
        ".cpu(",
        "tilize",
        "untilize",
        "reshard",
        "super().prefill_forward",
        "super().decode_forward",
    )
    for method in methods:
        source = inspect.getsource(method)
        for token in forbidden:
            assert token not in source, f"{method.__name__} contains forbidden runtime token {token!r}"

    mlp_source = inspect.getsource(FusedDecoder._mlp_forward)
    update_source = inspect.getsource(FusedDecoder._update_cache)
    decode_source = inspect.getsource(FusedDecoder.decode_forward)
    create_heads_source = inspect.getsource(FusedDecoder._create_decode_heads)
    concat_heads_source = inspect.getsource(FusedDecoder._concatenate_decode_heads)
    assert "input_tensor_a_activations=[ttnn.UnaryOpType.SILU]" in mlp_source
    assert "ttnn.silu(" not in mlp_source
    assert update_source.count("paged_fused_update_cache(") == 1
    assert "paged_update_cache(" not in update_source.replace("paged_fused_update_cache(", "")
    assert "scaled_dot_product_attention_decode(" in inspect.getsource(FusedDecoder._decode_attention)
    assert "nlp_create_qkv_heads_decode(" in create_heads_source
    assert "nlp_concat_heads_decode(" in concat_heads_source
    assert "ttnn.permute(" not in create_heads_source + decode_source
    assert concat_heads_source.count("ttnn.permute(") == 1


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_non_aligned_prefill_decode_cache_and_determinism(mesh_device):
    config = functional_test._config()
    state = functional_test._synthetic_state(config)
    decoder = _decoder(FusedDecoder, state, config, mesh_device)
    reference_layer = functional_test._hf_layer(state, config)

    for seq_len in (3, EMITTED_PREFILL_SEQUENCE, 33):
        generator = torch.Generator().manual_seed(1200 + seq_len)
        hidden = torch.randn(
            (1, EMITTED_BATCH, seq_len, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        reference, reference_key, reference_value, reference_cache = functional_test._reference_layer(
            reference_layer,
            hidden,
            config,
        )
        key_cache, value_cache = functional_test._empty_caches(config, mesh_device)
        actual = decoder.prefill_forward(functional_test._tt_tensor(hidden, mesh_device), key_cache, value_cache)
        actual_host = functional_test._to_host(actual)
        functional_test._assert_pcc(reference, actual_host, 0.995, f"fused prefill seq={seq_len}")
        functional_test._assert_pcc(
            reference_key,
            functional_test._to_host(key_cache)[:, :, :seq_len, :],
            0.99,
            f"fused prefill key seq={seq_len}",
        )
        functional_test._assert_pcc(
            reference_value,
            functional_test._to_host(value_cache)[:, :, :seq_len, :],
            0.99,
            f"fused prefill value seq={seq_len}",
        )

        if seq_len != EMITTED_PREFILL_SEQUENCE:
            continue

        second_key, second_value = functional_test._empty_caches(config, mesh_device)
        second = decoder.prefill_forward(functional_test._tt_tensor(hidden, mesh_device), second_key, second_value)
        assert torch.equal(actual_host, functional_test._to_host(second))
        assert torch.equal(functional_test._to_host(key_cache), functional_test._to_host(second_key))
        assert torch.equal(functional_test._to_host(value_cache), functional_test._to_host(second_value))

        decode_hidden = torch.randn(
            (1, EMITTED_BATCH, 1, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        reference_decode, reference_key, reference_value, _ = functional_test._reference_layer(
            reference_layer,
            decode_hidden,
            config,
            start_pos=seq_len,
            cache=reference_cache,
        )
        first_decode = decoder.decode_forward(
            functional_test._tt_tensor(decode_hidden, mesh_device),
            key_cache,
            value_cache,
            current_pos=seq_len,
        )
        second_decode = decoder.decode_forward(
            functional_test._tt_tensor(decode_hidden, mesh_device),
            second_key,
            second_value,
            current_pos=seq_len,
        )
        functional_test._assert_pcc(
            reference_decode,
            functional_test._to_host(first_decode),
            0.995,
            "fused decode output",
        )
        functional_test._assert_pcc(
            reference_key,
            functional_test._to_host(key_cache)[:, :, seq_len : seq_len + 1, :],
            0.99,
            "fused decode key append",
        )
        functional_test._assert_pcc(
            reference_value,
            functional_test._to_host(value_cache)[:, :, seq_len : seq_len + 1, :],
            0.99,
            "fused decode value append",
        )
        assert torch.equal(functional_test._to_host(first_decode), functional_test._to_host(second_decode))
        assert torch.equal(functional_test._to_host(key_cache), functional_test._to_host(second_key))
        assert torch.equal(functional_test._to_host(value_cache), functional_test._to_host(second_value))

    del decoder, reference_layer, state
    gc.collect()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_many_lengths_and_positions_keep_view_allocations_bounded(mesh_device):
    config = functional_test._config()
    state = functional_test._synthetic_state(config)
    decoder = _decoder(FusedDecoder, state, config, mesh_device)

    prefill_cycle = (3, 17, 33, 64, 127, 3)
    for seq_len in prefill_cycle:
        decoder._get_prefill_rotary_views(seq_len)
    prefill_allocations = []
    for _ in range(4):
        for seq_len in prefill_cycle:
            decoder._get_prefill_rotary_views(seq_len)
        prefill_allocations.append(_allocated_dram_bytes(mesh_device))
    assert max(prefill_allocations) == min(prefill_allocations)
    assert decoder.prefill_rotary_view[0] == 3

    for current_pos in range(EMITTED_CACHE_LENGTH):
        decoder._get_decode_position_views(current_pos)
    decoder._get_decode_position_views(0)
    decode_allocations = []
    for current_pos in range(1, EMITTED_CACHE_LENGTH):
        decoder._get_decode_position_views(current_pos)
        decode_allocations.append(_allocated_dram_bytes(mesh_device))
    decoder._get_decode_position_views(0)
    decode_allocations.append(_allocated_dram_bytes(mesh_device))
    assert max(decode_allocations) == min(decode_allocations)
    assert decoder.decode_position_view[0] == 0

    print(
        "bounded view cache: "
        f"prefill_min={min(prefill_allocations)}, prefill_max={max(prefill_allocations)}, "
        f"decode_min={min(decode_allocations)}, decode_max={max(decode_allocations)}"
    )


@pytest.mark.parametrize("layer_idx", [0, REPRESENTATIVE_LAYER, 63])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_real_weight_first_middle_last_layer(mesh_device, layer_idx):
    config = functional_test._config()
    with _functional_helpers_for_layer(layer_idx):
        state = functional_test._real_state()
        decoder = _decoder(FusedDecoder, state, config, mesh_device, layer_idx=layer_idx)
        functional_decoder = _decoder(FunctionalDecoder, state, config, mesh_device, layer_idx=layer_idx)
        reference_layer = functional_test._hf_layer(state, config)
        key_cache, value_cache = functional_test._empty_caches(config, mesh_device)
        functional_key_cache, functional_value_cache = functional_test._empty_caches(config, mesh_device)
        generator = torch.Generator().manual_seed(3200 + layer_idx)
        prefill_hidden = torch.randn(
            (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        reference, reference_key, reference_value, reference_cache = functional_test._reference_layer(
            reference_layer,
            prefill_hidden,
            config,
        )
        functional = functional_decoder.prefill_forward(
            functional_test._tt_tensor(prefill_hidden, mesh_device),
            functional_key_cache,
            functional_value_cache,
        )
        actual = decoder.prefill_forward(
            functional_test._tt_tensor(prefill_hidden, mesh_device),
            key_cache,
            value_cache,
        )
        _assert_preserves_functional_pcc(
            reference,
            functional_test._to_host(functional),
            functional_test._to_host(actual),
            0.995,
            f"real layer {layer_idx} prefill",
        )
        functional_test._assert_pcc(
            reference_key,
            functional_test._to_host(key_cache)[:, :, :EMITTED_PREFILL_SEQUENCE, :],
            0.99,
            f"real layer {layer_idx} key",
        )
        functional_test._assert_pcc(
            reference_value,
            functional_test._to_host(value_cache)[:, :, :EMITTED_PREFILL_SEQUENCE, :],
            0.99,
            f"real layer {layer_idx} value",
        )
        decode_hidden = torch.randn(
            (1, EMITTED_BATCH, 1, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        reference, reference_key, reference_value, _ = functional_test._reference_layer(
            reference_layer,
            decode_hidden,
            config,
            start_pos=EMITTED_PREFILL_SEQUENCE,
            cache=reference_cache,
        )
        functional = functional_decoder.decode_forward(
            functional_test._tt_tensor(decode_hidden, mesh_device),
            functional_key_cache,
            functional_value_cache,
            current_pos=EMITTED_PREFILL_SEQUENCE,
        )
        actual = decoder.decode_forward(
            functional_test._tt_tensor(decode_hidden, mesh_device),
            key_cache,
            value_cache,
            current_pos=EMITTED_PREFILL_SEQUENCE,
        )
        _assert_preserves_functional_pcc(
            reference,
            functional_test._to_host(functional),
            functional_test._to_host(actual),
            0.995,
            f"real layer {layer_idx} decode",
        )
        functional_test._assert_pcc(
            reference_key,
            functional_test._to_host(key_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1],
            0.99,
            f"real layer {layer_idx} decode key",
        )
        functional_test._assert_pcc(
            reference_value,
            functional_test._to_host(value_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1],
            0.99,
            f"real layer {layer_idx} decode value",
        )
    del decoder, functional_decoder, reference_layer, state
    gc.collect()


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_repeated_and_traced_decode_preserve_full_cache(mesh_device, device_params):
    config = functional_test._config()
    state = functional_test._synthetic_state(config)
    decoder = _decoder(FusedDecoder, state, config, mesh_device)
    reference_layer = functional_test._hf_layer(state, config)
    key_cache, value_cache = functional_test._empty_caches(config, mesh_device)
    generator = torch.Generator().manual_seed(3391)
    prefill = torch.randn(
        (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    _, _, _, reference_cache = functional_test._reference_layer(reference_layer, prefill, config)
    decoder.prefill_forward(functional_test._tt_tensor(prefill, mesh_device), key_cache, value_cache)

    for current_pos in range(EMITTED_PREFILL_SEQUENCE, EMITTED_PREFILL_SEQUENCE + 3):
        hidden = torch.randn(
            (1, EMITTED_BATCH, 1, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        reference, _, _, _ = functional_test._reference_layer(
            reference_layer,
            hidden,
            config,
            start_pos=current_pos,
            cache=reference_cache,
        )
        actual = decoder.decode_forward(
            functional_test._tt_tensor(hidden, mesh_device),
            key_cache,
            value_cache,
            current_pos=current_pos,
        )
        functional_test._assert_pcc(
            reference, functional_test._to_host(actual), 0.995, f"repeated decode {current_pos}"
        )
        functional_test._assert_pcc(
            reference_cache.layers[REPRESENTATIVE_LAYER].keys[:, :, : current_pos + 1, :],
            functional_test._to_host(key_cache)[:, :, : current_pos + 1, :],
            0.99,
            f"repeated key prefix {current_pos}",
        )
        functional_test._assert_pcc(
            reference_cache.layers[REPRESENTATIVE_LAYER].values[:, :, : current_pos + 1, :],
            functional_test._to_host(value_cache)[:, :, : current_pos + 1, :],
            0.99,
            f"repeated value prefix {current_pos}",
        )

    trace_pos = EMITTED_PREFILL_SEQUENCE + 3
    trace_hidden = torch.randn(
        (1, EMITTED_BATCH, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference, _, _, _ = functional_test._reference_layer(
        reference_layer,
        trace_hidden,
        config,
        start_pos=trace_pos,
        cache=reference_cache,
    )
    trace_input = functional_test._tt_tensor(trace_hidden, mesh_device)
    warm = decoder.decode_forward(trace_input, key_cache, value_cache, current_pos=trace_pos)
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = decoder.decode_forward(trace_input, key_cache, value_cache, current_pos=trace_pos)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        first_output = functional_test._to_host(trace_output)
        first_key = functional_test._to_host(key_cache)
        first_value = functional_test._to_host(value_cache)
        for _ in range(9):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        final_output = functional_test._to_host(trace_output)
        final_key = functional_test._to_host(key_cache)
        final_value = functional_test._to_host(value_cache)
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    functional_test._assert_pcc(reference, final_output, 0.995, "traced fused decode")
    assert torch.equal(first_output, final_output)
    assert torch.equal(first_key, final_key)
    assert torch.equal(first_value, final_value)
    del decoder, reference_layer, state
    gc.collect()


@pytest.mark.skipif(not os.getenv(CONTEXT_PROBE_ENV), reason="manual fused context-preservation gate")
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.timeout(600)
def test_fused_preserves_functional_context_capacity(mesh_device):
    seq_len = int(os.environ[CONTEXT_PROBE_ENV])
    config = functional_test._config()
    state = functional_test._synthetic_state(config)
    decoder = FusedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
        max_cache_len=seq_len,
    )
    key_cache, value_cache = functional_test._empty_caches(config, mesh_device, max_cache_len=seq_len)
    hidden = torch.zeros((1, EMITTED_BATCH, seq_len, config.hidden_size), dtype=torch.bfloat16)
    output = decoder.prefill_forward(
        functional_test._tt_tensor(hidden, mesh_device),
        key_cache,
        value_cache,
    )
    host = functional_test._to_host(output)
    assert tuple(host.shape) == tuple(hidden.shape)
    print(f"fused capacity PASS: batch={EMITTED_BATCH}, seq_len={seq_len}, output_shape={tuple(host.shape)}")


def _measure_decoder(decoder_cls, state, config, mesh_device, prefill_host, decode_host):
    decoder = _decoder(decoder_cls, state, config, mesh_device)
    key_cache, value_cache = functional_test._empty_caches(config, mesh_device)
    prefill_input = functional_test._tt_tensor(prefill_host, mesh_device)
    decode_input = functional_test._tt_tensor(decode_host, mesh_device)

    warm = decoder.prefill_forward(prefill_input, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    prefill_output_host = functional_test._to_host(warm)
    warm.deallocate(True)
    prefill_samples = []
    for _ in range(int(os.getenv("QWEN3_32B_FUSED_PREFILL_ITERATIONS", "7"))):
        start = time.perf_counter()
        output = decoder.prefill_forward(prefill_input, key_cache, value_cache)
        ttnn.synchronize_device(mesh_device)
        prefill_samples.append((time.perf_counter() - start) * 1000.0)
        output.deallocate(True)

    warm = decoder.decode_forward(
        decode_input,
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = decoder.decode_forward(
        decode_input,
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    iterations = int(os.getenv("QWEN3_32B_FUSED_DECODE_ITERATIONS", "50"))
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(iterations):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        decode_ms = (time.perf_counter() - start) * 1000.0 / iterations
        output_host = functional_test._to_host(trace_output)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        second_output_host = functional_test._to_host(trace_output)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    assert torch.equal(output_host, second_output_host)
    result = {
        "prefill_median_ms": statistics.median(prefill_samples),
        "prefill_min_ms": min(prefill_samples),
        "prefill_samples_ms": prefill_samples,
        "traced_decode_mean_ms": decode_ms,
        "traced_decode_iterations": iterations,
        "prefill_output": prefill_output_host,
        "decode_output": output_host,
    }
    del decoder
    gc.collect()
    return result


@pytest.mark.skipif(os.getenv(RUN_PERF_ENV) != "1", reason="manual real-weight fused performance gate")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.timeout(1200)
def test_fused_beats_functional_warmed_prefill_and_traced_decode(mesh_device, device_params):
    config = functional_test._config()
    state = functional_test._real_state()
    reference_layer = functional_test._hf_layer(state, config)
    generator = torch.Generator().manual_seed(4401)
    prefill_host = torch.randn(
        (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    decode_host = torch.randn(
        (1, EMITTED_BATCH, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference_prefill, _, _, reference_cache = functional_test._reference_layer(reference_layer, prefill_host, config)
    reference_decode, _, _, _ = functional_test._reference_layer(
        reference_layer,
        decode_host,
        config,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )

    results = {}
    for name, decoder_cls in (("functional", FunctionalDecoder), ("fused", FusedDecoder)):
        measured = _measure_decoder(decoder_cls, state, config, mesh_device, prefill_host, decode_host)
        prefill_pcc = functional_test._assert_pcc(
            reference_prefill,
            measured.pop("prefill_output"),
            0.995,
            f"{name} warmed prefill",
        )
        decode_pcc = functional_test._assert_pcc(
            reference_decode,
            measured.pop("decode_output"),
            0.995,
            f"{name} traced decode",
        )
        measured["prefill_pcc"] = prefill_pcc
        measured["decode_pcc"] = decode_pcc
        results[name] = measured
        print(f"PERF_RESULT {name}={measured}")

    assert results["fused"]["prefill_median_ms"] < results["functional"]["prefill_median_ms"]
    assert results["fused"]["traced_decode_mean_ms"] < results["functional"]["traced_decode_mean_ms"]
    output_dir = Path(os.getenv(RESULTS_DIR_ENV, "models/autoports/qwen_qwen3_32b/doc/fused_decoder/results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "before_after.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3-32B",
                "layer": REPRESENTATIVE_LAYER,
                "batch": EMITTED_BATCH,
                "prefill_sequence": EMITTED_PREFILL_SEQUENCE,
                "precision": "BF16 weights/activations/cache",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    del reference_layer, state
    gc.collect()
