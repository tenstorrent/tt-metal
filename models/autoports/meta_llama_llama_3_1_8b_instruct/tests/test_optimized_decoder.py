# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import inspect
import os
import time

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    LAYER_IDX,
    _assert_pcc,
    _config,
    _hf_layer,
    _real_state,
    _reference_layer,
    _to_host,
    _tt_tensor,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import FunctionalDecoder
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import OptimizationConfig, OptimizedDecoder


def _empty_caches(config, mesh_device, *, batch=EMITTED_BATCH, dtype=ttnn.bfloat16):
    head_dim = config.hidden_size // config.num_attention_heads
    shape = (batch, config.num_key_value_heads, EMITTED_CACHE_LENGTH, head_dim)
    key = _tt_tensor(torch.zeros(shape, dtype=torch.bfloat16), mesh_device, dtype=dtype)
    value = _tt_tensor(torch.zeros(shape, dtype=torch.bfloat16), mesh_device, dtype=dtype)
    return key, value


def _policy(name: str) -> OptimizationConfig:
    dram_candidate = OptimizationConfig(
        attention_weight_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
        gate_up_weight_dtype=ttnn.bfloat4_b,
        gate_up_math_fidelity=ttnn.MathFidelity.LoFi,
        down_weight_dtype=ttnn.bfloat8_b,
        down_math_fidelity=ttnn.MathFidelity.HiFi2,
        decode_matmul_strategy="dram_sharded",
    )
    advisor_candidate = dram_candidate.with_changes(decode_matmul_strategy="advisor_1d")
    final_candidate = OptimizationConfig()
    policies = {
        "optimized": final_candidate,
        "advisor_1d": advisor_candidate,
        "attn_bfp8_lofi": dram_candidate.with_changes(attention_math_fidelity=ttnn.MathFidelity.LoFi),
        "attn_bfp4_lofi": dram_candidate.with_changes(
            attention_weight_dtype=ttnn.bfloat4_b,
            attention_math_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "attn_bfp4_hifi2": dram_candidate.with_changes(attention_weight_dtype=ttnn.bfloat4_b),
        "down_bfp4_lofi": dram_candidate.with_changes(
            down_weight_dtype=ttnn.bfloat4_b,
            down_math_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "down_bfp4_hifi2": dram_candidate.with_changes(down_weight_dtype=ttnn.bfloat4_b),
        "advisor_down_bfp4": advisor_candidate.with_changes(
            down_weight_dtype=ttnn.bfloat4_b,
            down_math_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "advisor_all_bfp4": final_candidate,
        "advisor_exact_chain": final_candidate.with_changes(advisor_exact_residual_chain=True),
        "final_default_sdpa": final_candidate.with_changes(explicit_sdpa_program_config=False),
        "final_prefill_grid10": final_candidate.with_changes(prefill_grid=(10, 10)),
        "final_prefill_grid11": final_candidate.with_changes(prefill_grid=(11, 10)),
        "final_sdpa_grid10": final_candidate.with_changes(sdpa_grid=(10, 10)),
        "final_sdpa_grid11": final_candidate.with_changes(sdpa_grid=(11, 10)),
        "final_gate_up_hifi2": final_candidate.with_changes(gate_up_math_fidelity=ttnn.MathFidelity.HiFi2),
        "final_gate_up_subblock5": final_candidate.with_changes(advisor_gate_up_out_subblock_w=5),
        "all_bfp8_hifi2": dram_candidate.with_changes(
            attention_weight_dtype=ttnn.bfloat8_b,
            attention_math_fidelity=ttnn.MathFidelity.HiFi2,
            gate_up_weight_dtype=ttnn.bfloat8_b,
            gate_up_math_fidelity=ttnn.MathFidelity.HiFi2,
            down_weight_dtype=ttnn.bfloat8_b,
            down_math_fidelity=ttnn.MathFidelity.HiFi2,
        ),
        "all_bfp8_hifi2_grid10": dram_candidate.with_changes(
            gate_up_weight_dtype=ttnn.bfloat8_b,
            gate_up_math_fidelity=ttnn.MathFidelity.HiFi2,
            prefill_grid=(10, 10),
        ),
        "packed_gate_up": dram_candidate.with_changes(packed_gate_up=True),
        "packed_gate_up_grid11": dram_candidate.with_changes(
            packed_gate_up=True,
            prefill_grid=(11, 10),
        ),
        "down_bfp4_geometry_16": dram_candidate.with_changes(
            down_weight_dtype=ttnn.bfloat4_b,
            down_math_fidelity=ttnn.MathFidelity.LoFi,
            qkv_cores=16,
            output_cores=16,
            gate_up_cores=16,
            down_cores=16,
        ),
        "down_bfp4_geometry_8": dram_candidate.with_changes(
            down_weight_dtype=ttnn.bfloat4_b,
            down_math_fidelity=ttnn.MathFidelity.LoFi,
            qkv_cores=8,
            output_cores=8,
            gate_up_cores=8,
            down_cores=8,
        ),
        "down_bfp4_down_16": dram_candidate.with_changes(
            down_weight_dtype=ttnn.bfloat4_b,
            down_math_fidelity=ttnn.MathFidelity.LoFi,
            down_cores=16,
        ),
        "down_bfp4_down_8": dram_candidate.with_changes(
            down_weight_dtype=ttnn.bfloat4_b,
            down_math_fidelity=ttnn.MathFidelity.LoFi,
            down_cores=8,
        ),
        "down_bfp4_gate_up_16": dram_candidate.with_changes(
            down_weight_dtype=ttnn.bfloat4_b,
            down_math_fidelity=ttnn.MathFidelity.LoFi,
            gate_up_cores=16,
        ),
        "down_bfp4_gate_up_8": dram_candidate.with_changes(
            down_weight_dtype=ttnn.bfloat4_b,
            down_math_fidelity=ttnn.MathFidelity.LoFi,
            gate_up_cores=8,
        ),
        "down_bfp4_gate_up_8_block8": dram_candidate.with_changes(
            down_weight_dtype=ttnn.bfloat4_b,
            down_math_fidelity=ttnn.MathFidelity.LoFi,
            gate_up_cores=8,
            gate_up_in0_block_w=8,
        ),
        "geometry_16": dram_candidate.with_changes(
            qkv_cores=16,
            output_cores=16,
            gate_up_cores=16,
            down_cores=16,
        ),
        "geometry_8": dram_candidate.with_changes(
            qkv_cores=8,
            output_cores=8,
            gate_up_cores=8,
            down_cores=8,
        ),
    }
    try:
        return policies[name]
    except KeyError as error:
        raise ValueError(f"Unknown optimized-decoder policy {name!r}; choices={tuple(policies)}") from error


def _decoder(variant, state, config, mesh_device, *, batch=EMITTED_BATCH):
    if variant == "functional":
        return FunctionalDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=LAYER_IDX,
            mesh_device=mesh_device,
            batch=batch,
        )
    return OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        optimization_config=_policy(variant),
    )


def test_runtime_path_is_optimized_and_host_fallback_free():
    assert OptimizedDecoder.prefill_forward.__module__.endswith("optimized_decoder")
    assert OptimizedDecoder.decode_forward.__module__.endswith("optimized_decoder")
    assert OptimizedDecoder._mlp_decode.__module__.endswith("optimized_decoder")
    for method in (
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder.decode_forward,
        OptimizedDecoder._mlp_prefill,
        OptimizedDecoder._mlp_decode,
    ):
        source = inspect.getsource(method)
        for token in ("super().prefill", "super().decode", "from_torch", "to_torch", "torch."):
            assert token not in source, f"{method.__name__} contains forbidden runtime token {token!r}"


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_real_weight_prefill_decode_and_cache_contract(mesh_device):
    config = _config()
    state = _real_state()
    decoder = _decoder("optimized", state, config, mesh_device)
    reference_layer = _hf_layer(state, config)
    key_cache, value_cache = _empty_caches(config, mesh_device)

    generator = torch.Generator().manual_seed(31)
    prefill_hidden = torch.randn(
        (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference_prefill, _, _, reference_cache = _reference_layer(reference_layer, prefill_hidden, config)
    actual_prefill = decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache)
    _assert_pcc(reference_prefill, _to_host(actual_prefill), 0.99, "optimized real prefill seq=18")

    decode_hidden = torch.randn(
        (1, EMITTED_BATCH, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference_decode, decode_key, decode_value, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    actual_decode = decoder.decode_forward(
        _tt_tensor(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    _assert_pcc(reference_decode, _to_host(actual_decode), 0.99, "optimized real decode pos=18")
    cache_slice = slice(EMITTED_PREFILL_SEQUENCE, EMITTED_PREFILL_SEQUENCE + 1)
    _assert_pcc(decode_key, _to_host(key_cache)[:, :, cache_slice, :], 0.99, "optimized key append")
    _assert_pcc(decode_value, _to_host(value_cache)[:, :, cache_slice, :], 0.99, "optimized value append")

    del decoder, reference_layer, state
    gc.collect()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_non_aligned_sequence_batch_and_repeated_run_determinism(mesh_device):
    config = _config()
    state = _real_state()
    batch = 13
    seq_len = 7
    decoder = _decoder("optimized", state, config, mesh_device, batch=batch)
    reference_layer = _hf_layer(state, config)
    generator = torch.Generator().manual_seed(713)
    prefill_hidden = torch.randn((1, batch, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    decode_hidden = torch.randn((1, batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    reference_prefill, _, _, reference_cache = _reference_layer(reference_layer, prefill_hidden, config)
    reference_decode, _, _, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=seq_len,
        cache=reference_cache,
    )

    outputs = []
    for run in range(3):
        key_cache, value_cache = _empty_caches(config, mesh_device, batch=batch)
        actual_prefill = decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache)
        actual_decode = decoder.decode_forward(
            _tt_tensor(decode_hidden, mesh_device),
            key_cache,
            value_cache,
            current_pos=seq_len,
        )
        prefill_host = _to_host(actual_prefill)
        decode_host = _to_host(actual_decode)
        _assert_pcc(reference_prefill, prefill_host, 0.99, f"nonaligned run={run} prefill")
        _assert_pcc(reference_decode, decode_host, 0.99, f"nonaligned run={run} decode")
        outputs.append((prefill_host, decode_host))

    for output in outputs[1:]:
        assert torch.equal(outputs[0][0], output[0])
        assert torch.equal(outputs[0][1], output[1])

    del decoder, reference_layer, state
    gc.collect()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_bfp8_cache_candidate_real_pcc(mesh_device):
    config = _config()
    state = _real_state()
    decoder = _decoder("optimized", state, config, mesh_device)
    reference_layer = _hf_layer(state, config)
    key_cache, value_cache = _empty_caches(config, mesh_device, dtype=ttnn.bfloat8_b)
    generator = torch.Generator().manual_seed(88)
    prefill_hidden = torch.randn(
        (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    _, _, _, reference_cache = _reference_layer(reference_layer, prefill_hidden, config)
    decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache)
    decode_hidden = torch.randn((1, EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    reference_decode, _, _, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    actual_decode = decoder.decode_forward(
        _tt_tensor(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    _assert_pcc(reference_decode, _to_host(actual_decode), 0.99, "BFP8 KV-cache real decode")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_candidate_executes_real_optimized_path(mesh_device):
    if os.environ.get("RUN_OPTIMIZED_DECODER_CANDIDATES") != "1":
        pytest.skip("Set RUN_OPTIMIZED_DECODER_CANDIDATES=1 for candidate coverage")
    variant = os.environ.get("OPTIMIZED_DECODER_CANDIDATE_VARIANT", "advisor_1d")
    config = _config()
    state = _real_state()
    decoder = _decoder(variant, state, config, mesh_device)
    reference_layer = _hf_layer(state, config)
    key_cache, value_cache = _empty_caches(config, mesh_device)
    generator = torch.Generator().manual_seed(319)
    prefill_hidden = torch.randn(
        (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference_prefill, _, _, reference_cache = _reference_layer(reference_layer, prefill_hidden, config)
    actual_prefill = decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache)
    _assert_pcc(reference_prefill, _to_host(actual_prefill), 0.99, f"{variant} real prefill")
    decode_hidden = torch.randn((1, EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    reference_decode, _, _, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    actual_decode = decoder.decode_forward(
        _tt_tensor(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    _assert_pcc(reference_decode, _to_host(actual_decode), 0.99, f"{variant} real decode")
    assert decoder.use_advisor_1d == (_policy(variant).decode_matmul_strategy == "advisor_1d")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_real_weight_traced_decode_replay_correctness_and_input_refresh(mesh_device):
    config = _config()
    state = _real_state()
    decoder = _decoder("optimized", state, config, mesh_device)
    reference_layer = _hf_layer(state, config)
    key_cache, value_cache = _empty_caches(config, mesh_device)
    generator = torch.Generator().manual_seed(444)
    prefill_hidden = torch.randn(
        (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    _, _, _, reference_cache = _reference_layer(reference_layer, prefill_hidden, config)
    decoder.prefill_forward(
        _tt_tensor(prefill_hidden, mesh_device),
        key_cache,
        value_cache,
    )

    decode_hidden = torch.randn(
        (1, EMITTED_BATCH, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference_decode, reference_key, reference_value, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    hidden = _tt_tensor(decode_hidden, mesh_device)
    eager_output = decoder.decode_forward(
        hidden,
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    _assert_pcc(reference_decode, _to_host(eager_output), 0.99, "trace eager control")
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        hidden,
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    # Capture records the commands and persistent output address; only a replay
    # executes those commands and produces a valid output at that address.
    replay_outputs = []
    try:
        for _ in range(5):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            replay_outputs.append(_to_host(traced_output))

        for replay_idx, output in enumerate(replay_outputs):
            _assert_pcc(reference_decode, output, 0.99, f"real trace replay={replay_idx}")
        for output in replay_outputs[1:]:
            assert torch.equal(replay_outputs[0], output)
        cache_slice = slice(EMITTED_PREFILL_SEQUENCE, EMITTED_PREFILL_SEQUENCE + 1)
        _assert_pcc(
            reference_key,
            _to_host(key_cache)[:, :, cache_slice, :],
            0.99,
            "real trace key append",
        )
        _assert_pcc(
            reference_value,
            _to_host(value_cache)[:, :, cache_slice, :],
            0.99,
            "real trace value append",
        )

        refreshed_hidden = torch.randn(
            (1, EMITTED_BATCH, 1, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        _, _, _, refreshed_reference_cache = _reference_layer(reference_layer, prefill_hidden, config)
        refreshed_reference, refreshed_key, refreshed_value, _ = _reference_layer(
            reference_layer,
            refreshed_hidden,
            config,
            start_pos=EMITTED_PREFILL_SEQUENCE,
            cache=refreshed_reference_cache,
        )
        refreshed_host_tensor = ttnn.from_torch(
            refreshed_hidden,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        ttnn.copy_host_to_device_tensor(refreshed_host_tensor, hidden)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        refreshed_output = _to_host(traced_output)
        _assert_pcc(refreshed_reference, refreshed_output, 0.99, "refreshed real trace")
        assert not torch.equal(replay_outputs[0], refreshed_output)
        _assert_pcc(
            refreshed_key,
            _to_host(key_cache)[:, :, cache_slice, :],
            0.99,
            "refreshed trace key append",
        )
        _assert_pcc(
            refreshed_value,
            _to_host(value_cache)[:, :, cache_slice, :],
            0.99,
            "refreshed trace value append",
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_optimized_decoder_perf(mesh_device):
    if os.environ.get("RUN_OPTIMIZED_DECODER_PERF") != "1":
        pytest.skip("Set RUN_OPTIMIZED_DECODER_PERF=1 for warmed performance measurement")
    variant = os.environ.get("OPTIMIZED_DECODER_PERF_VARIANT", "optimized")
    cache_dtype = ttnn.bfloat8_b if os.environ.get("OPTIMIZED_DECODER_CACHE_DTYPE", "bf16") == "bfp8" else ttnn.bfloat16
    replay_count = int(os.environ.get("OPTIMIZED_DECODER_TRACE_REPLAYS", "20"))
    prefill_repeat_count = int(os.environ.get("OPTIMIZED_DECODER_PREFILL_REPEATS", "1"))
    batch = int(os.environ.get("OPTIMIZED_DECODER_BATCH", str(EMITTED_BATCH)))
    config = _config()
    state = _real_state()
    decoder = _decoder(variant, state, config, mesh_device, batch=batch)
    key_cache, value_cache = _empty_caches(config, mesh_device, batch=batch, dtype=cache_dtype)
    generator = torch.Generator().manual_seed(2026)
    prefill_hidden = _tt_tensor(
        torch.randn(
            (1, batch, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        ),
        mesh_device,
    )
    decode_hidden = _tt_tensor(
        torch.randn((1, batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16),
        mesh_device,
    )

    decoder.prefill_forward(prefill_hidden, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    signpost(header="PERF_PREFILL")
    start = time.perf_counter()
    for _ in range(prefill_repeat_count):
        decoder.prefill_forward(prefill_hidden, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    prefill_ms = (time.perf_counter() - start) * 1000.0 / prefill_repeat_count
    signpost(header="PERF_PREFILL_END")

    decoder.decode_forward(
        decode_hidden,
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        decode_hidden,
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        signpost(header="PERF_DECODE")
        start = time.perf_counter()
        for _ in range(replay_count):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        decode_ms = (time.perf_counter() - start) * 1000.0 / replay_count
        signpost(header="PERF_DECODE_END")
        assert tuple(traced_output.shape) == (1, batch, 1, config.hidden_size)
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    print(
        "PERF_RESULT "
        f"variant={variant} cache_dtype={cache_dtype} batch={batch} "
        f"seq={EMITTED_PREFILL_SEQUENCE} prefill_ms={prefill_ms:.6f} "
        f"traced_decode_ms={decode_ms:.6f} prefill_repeats={prefill_repeat_count} "
        f"trace_replays={replay_count}"
    )
