# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness, trace, and performance coverage for optimized Gemma-4 decode."""

from __future__ import annotations

import inspect
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import test_functional_decoder as functional_tests
from models.autoports.google_gemma_4_26b_a4b_it.tests.test_fused_decoder import (
    _load_layer_state,
    _load_text_config,
    _make_perf_args,
    _measure_warmed,
    _to_torch,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import (
    POLICIES,
    SPARSE_SINGLE_CORE_MCAST_BLOCKER,
    OptimizedDecoder,
    _sparse_program_config,
    sparse_geometry_host_rejection,
)
from models.common.utility_functions import comp_pcc

ARTIFACT_DIR = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder")


@contextmanager
def _optimized_acceptance_oracle(candidate=None):
    original_decoder = functional_tests.FunctionalDecoder
    original_artifact_dir = functional_tests.ARTIFACT_DIR
    original_candidate = OptimizedDecoder.DEFAULT_CANDIDATE
    functional_tests.FunctionalDecoder = OptimizedDecoder
    functional_tests.ARTIFACT_DIR = ARTIFACT_DIR
    if candidate is not None:
        OptimizedDecoder.DEFAULT_CANDIDATE = candidate
    try:
        yield
    finally:
        functional_tests.FunctionalDecoder = original_decoder
        functional_tests.ARTIFACT_DIR = original_artifact_dir
        OptimizedDecoder.DEFAULT_CANDIDATE = original_candidate


def test_optimized_hot_path_is_owned():
    assert OptimizedDecoder is not FusedDecoder
    assert OptimizedDecoder._moe_decode_single_user is not FusedDecoder._moe_decode_single_user
    assert OptimizedDecoder._dense_mlp is not FusedDecoder._dense_mlp
    assert OptimizedDecoder._moe_prefill_tile is not FusedDecoder._moe_prefill_tile
    assert OptimizedDecoder.decode_forward is not FusedDecoder.decode_forward
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            OptimizedDecoder.decode_forward,
            OptimizedDecoder._moe_decode_single_user,
            OptimizedDecoder._dense_mlp,
            OptimizedDecoder._moe_prefill_tile,
            OptimizedDecoder._fill_prefill_cache,
        )
    )
    for forbidden in ("torch.", "from_torch", "to_torch"):
        assert forbidden not in source
    assert "sparse_matmul" in source
    assert "compute_kernel_config" in source


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_hf_acceptance(mesh_device, device_params, layer_idx):
    with _optimized_acceptance_oracle():
        functional_tests.test_functional_decoder_real_weights_prefill_decode(
            mesh_device,
            device_params,
            layer_idx,
            layer_idx == 0,
            0.995,
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize(
    "candidate",
    [
        "bfp4_experts_isolated",
        "bfp8_experts_hifi4",
        "bfp8_experts_lofi",
        "bfp4_attention_only",
        "bfp8_attention_only",
        "bfp4_dense_gate_up",
        "bfp4_dense_all",
        "bfp4_dense_gate_up_packed",
        "bfp8_projection_hifi2",
    ],
)
def test_optimized_hf_precision_isolation(mesh_device, device_params, layer_idx, candidate):
    if os.getenv("GEMMA4_OPTIMIZED_CANDIDATES") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_CANDIDATES=1 to run real-weight HF isolation")
    with _optimized_acceptance_oracle(candidate):
        functional_tests.test_functional_decoder_real_weights_prefill_decode(
            mesh_device,
            device_params,
            layer_idx,
            layer_idx == 0,
            0.995,
        )
    layer_type = _load_text_config().layer_types[layer_idx]
    generic = ARTIFACT_DIR / f"pcc_layer{layer_idx}_{layer_type}_shared{int(layer_idx == 0)}.json"
    if generic.exists():
        artifact = json.loads(generic.read_text())
        artifact["candidate"] = candidate
        (ARTIFACT_DIR / f"pcc_{candidate}_layer{layer_idx}_{layer_type}.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n"
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_traced_decode_batch1(mesh_device, device_params, layer_idx):
    with _optimized_acceptance_oracle():
        functional_tests.test_traced_decode_batch_contract(mesh_device, device_params, layer_idx, 1)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_bfp8_kv_cache_candidate(mesh_device, device_params, layer_idx):
    if os.getenv("GEMMA4_OPTIMIZED_CACHE") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_CACHE=1 to run the reduced-cache trial")
    cfg = _load_text_config()
    state = _load_layer_state(layer_idx)
    layer_type = cfg.layer_types[layer_idx]
    bf16_decoder = OptimizedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    bfp8_decoder = OptimizedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    bf16_args = _make_perf_args(bf16_decoder, mesh_device, layer_type=layer_type, seq_len=1024, batch=1)
    bfp8_args = _make_perf_args(bfp8_decoder, mesh_device, layer_type=layer_type, seq_len=1024, batch=1)
    reduced_cache = tuple(
        ttnn.typecast(cache, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for cache in bfp8_args[0]["kv_cache"]
    )
    bfp8_args[0]["kv_cache"] = reduced_cache
    bfp8_args[1]["kv_cache"] = reduced_cache
    bf16_decoder.prefill_forward(**bf16_args[0])
    bfp8_decoder.prefill_forward(**bfp8_args[0])
    bf16_output = bf16_decoder.decode_forward(**bf16_args[1])
    bfp8_output = bfp8_decoder.decode_forward(**bfp8_args[1])
    passed, pcc = comp_pcc(_to_torch(mesh_device, bf16_output), _to_torch(mesh_device, bfp8_output), 0.995)
    perf = _measure_warmed(
        bfp8_decoder,
        mesh_device,
        prefill_args=bfp8_args[0],
        decode_args=bfp8_args[1],
        batch=1,
    )
    artifact = {
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "logical_batch": 1,
        "sequence_length": 1024,
        "cache_dtype": "BFLOAT8_B",
        "cache_layout": "DRAM_INTERLEAVED",
        "bf16_vs_bfp8_decode_pcc": float(pcc),
        "result": perf,
    }
    (ARTIFACT_DIR / f"kv_bfp8_layer{layer_idx}_{layer_type}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    assert passed, pcc


def _dram_sharded_projection_configs(mesh_device, k, n, in0_block_w):
    dram_size = mesh_device.dram_grid_size()
    assert dram_size.y == 1
    cores = dram_size.x
    assert k % (32 * cores) == 0 and n % (32 * cores) == 0
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))})
    weight = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (k, n // cores), ttnn.ShardOrientation.ROW_MAJOR),
    )
    activation = ttnn.create_sharded_memory_config(
        shape=(32, k // cores),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    output = ttnn.create_sharded_memory_config(
        shape=(32, n // cores),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=1,
        per_core_N=n // (32 * cores),
        fused_activation=None,
    )
    return weight, activation, output, program


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("in0_block_w", [1, 11], ids=["kblock1", "kblock11"])
def test_optimized_dram_sharded_qkv_candidate(mesh_device, device_params, layer_idx, in0_block_w):
    if os.getenv("GEMMA4_OPTIMIZED_DRAM_SHARDED") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_DRAM_SHARDED=1 to run the QKV projection sweep")
    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    decoder = OptimizedDecoder.from_state_dict(
        _load_layer_state(layer_idx), hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device
    )
    _, decode_args = _make_perf_args(decoder, mesh_device, layer_type=layer_type, seq_len=1024, batch=1)
    x = decoder._rms_norm(decode_args["hidden_states"], decoder.weights.input_ln)
    reference = ttnn.linear(
        x, decoder.weights.qkv, dtype=decoder.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    n = decoder.layer_kind.qkv_width
    weight_mem, input_mem, output_mem, program = _dram_sharded_projection_configs(mesh_device, 2816, n, in0_block_w)
    weight = ttnn.to_memory_config(decoder.weights.qkv, weight_mem, dtype=decoder.weights.qkv.dtype)

    def candidate():
        sharded_x = ttnn.to_memory_config(x, input_mem, dtype=x.dtype)
        sharded_out = ttnn.linear(
            sharded_x,
            weight,
            dtype=decoder.activation_dtype,
            memory_config=output_mem,
            program_config=program,
            compute_kernel_config=decoder.attention_compute_config,
        )
        return ttnn.sharded_to_interleaved(sharded_out, ttnn.DRAM_MEMORY_CONFIG)

    try:
        output = candidate()
    except RuntimeError as error:
        assert in0_block_w == 11 and "beyond max L1 size" in str(error)
        artifact = {
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "logical_batch": 1,
            "tile_padded_rows": 32,
            "dram_cores": mesh_device.dram_grid_size().x,
            "input_shard_shape": [32, 2816 // mesh_device.dram_grid_size().x],
            "weight_shard_shape": [2816, n // mesh_device.dram_grid_size().x],
            "output_shard_shape": [32, n // mesh_device.dram_grid_size().x],
            "in0_block_w": in0_block_w,
            "status": "l1_capacity_rejected",
            "error": str(error).split("backtrace:", 1)[0].strip(),
        }
        (ARTIFACT_DIR / f"dram_qkv_layer{layer_idx}_{layer_type}_k{in0_block_w}.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n"
        )
        return
    passed, pcc = comp_pcc(_to_torch(mesh_device, reference), _to_torch(mesh_device, output), 0.995)
    candidate()
    ttnn.synchronize_device(mesh_device)
    samples = []
    reference_samples = []
    for _ in range(21):
        start = time.perf_counter()
        ttnn.linear(x, decoder.weights.qkv, dtype=decoder.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(mesh_device)
        reference_samples.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        candidate()
        ttnn.synchronize_device(mesh_device)
        samples.append((time.perf_counter() - start) * 1000)
    artifact = {
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "logical_batch": 1,
        "tile_padded_rows": 32,
        "weight_dtype": str(weight.dtype),
        "dram_cores": mesh_device.dram_grid_size().x,
        "input_shard_shape": [32, 2816 // mesh_device.dram_grid_size().x],
        "weight_shard_shape": [2816, n // mesh_device.dram_grid_size().x],
        "output_shard_shape": [32, n // mesh_device.dram_grid_size().x],
        "in0_block_w": in0_block_w,
        "pcc": float(pcc),
        "interleaved_host_ms_median": statistics.median(reference_samples),
        "interleaved_host_ms_samples": reference_samples,
        "boundary_inclusive_host_ms_median": statistics.median(samples),
        "boundary_inclusive_host_ms_samples": samples,
        "speedup_vs_interleaved": statistics.median(reference_samples) / statistics.median(samples),
        "status": "passed",
    }
    (ARTIFACT_DIR / f"dram_qkv_layer{layer_idx}_{layer_type}_k{in0_block_w}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    assert passed, pcc


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_sdpa_config_sweep(mesh_device, device_params, layer_idx):
    if os.getenv("GEMMA4_OPTIMIZED_SDPA") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_SDPA=1 to run the batch-1 SDPA sweep")
    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    decoder = OptimizedDecoder.from_state_dict(
        _load_layer_state(layer_idx), hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device
    )
    _, decode_args = _make_perf_args(decoder, mesh_device, layer_type=layer_type, seq_len=1024, batch=1)
    candidates = [
        ("grid8x4_q32_k64", 8, 4, 32, 64),
        ("grid8x8_q32_k64", 8, 8, 32, 64),
        ("grid8x8_q32_k128", 8, 8, 32, 128),
    ]
    reference = None
    rows = []
    for name, gx, gy, q_chunk, k_chunk in candidates:
        decoder.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )
        try:
            eager, traced, samples = _measure_decode_trace(decoder, mesh_device, decode_args)
            eager_torch = _to_torch(mesh_device, eager)
            traced_torch = _to_torch(mesh_device, traced)
            if reference is None:
                reference = eager_torch
            correct, pcc = comp_pcc(reference, eager_torch, 0.995)
            replay_correct, replay_pcc = comp_pcc(eager_torch, traced_torch, 0.999)
            rows.append(
                {
                    "candidate": name,
                    "grid": [gx, gy],
                    "q_chunk_size": q_chunk,
                    "k_chunk_size": k_chunk,
                    "whole_layer_trace_host_ms_median": statistics.median(samples),
                    "whole_layer_trace_host_ms_samples": samples,
                    "correctness_pcc": float(pcc),
                    "trace_replay_pcc": float(replay_pcc),
                    "status": "passed" if correct and replay_correct else "failed_pcc",
                }
            )
        except RuntimeError as error:
            rows.append({"candidate": name, "status": "runtime_error", "error": str(error)})
    passed = [row for row in rows if row["status"] == "passed"]
    artifact = {
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "logical_batch": 1,
        "sequence_length": 1024,
        "rows": rows,
        "winner": min(passed, key=lambda row: row["whole_layer_trace_host_ms_median"])["candidate"],
    }
    (ARTIFACT_DIR / f"sdpa_layer{layer_idx}_{layer_type}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    assert len(passed) >= 2, artifact


def _width_sharded_residual_config(cores):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))})
    shard_width = 2816 // cores
    memory = ttnn.create_sharded_memory_config(
        shape=(32, shard_width),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    block_w = shard_width // 32
    subblock_w = min(4, block_w)
    while block_w % subblock_w:
        subblock_w -= 1
    program = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[cores, 1],
        subblock_w=subblock_w,
        block_h=1,
        block_w=block_w,
        inplace=False,
    )
    return memory, program


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_width_sharded_residual_chain(mesh_device, device_params, layer_idx):
    if os.getenv("GEMMA4_OPTIMIZED_RESIDUAL") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_RESIDUAL=1 to run the residual-chain sweep")
    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    decoder = OptimizedDecoder.from_state_dict(
        _load_layer_state(layer_idx), hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device
    )
    _, args = _make_perf_args(decoder, mesh_device, layer_type=layer_type, seq_len=1024, batch=1)
    reference = decoder.decode_forward(**args)
    reference_torch = _to_torch(mesh_device, reference)
    rows = []
    for cores in (4, 8):
        memory, norm_program = _width_sharded_residual_config(cores)

        def norm(x, weight):
            return ttnn.rms_norm(
                x,
                epsilon=decoder.eps,
                weight=weight,
                compute_kernel_config=decoder.correctness_compute_config,
                memory_config=memory,
                program_config=norm_program,
            )

        def interleaved(x):
            return ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

        def sharded(x):
            return ttnn.to_memory_config(x, memory, dtype=x.dtype)

        def candidate():
            residual = sharded(args["hidden_states"])
            attn_in = interleaved(norm(residual, decoder.weights.input_ln))
            attn_out = decoder._attention_decode(
                attn_in,
                position_cos=args["position_cos"],
                position_sin=args["position_sin"],
                current_pos=args["current_pos"],
                page_table=args["page_table"],
                kv_cache=args["kv_cache"],
                cache_position_modulo=None,
            )
            hidden = ttnn.add(residual, norm(sharded(attn_out), decoder.weights.post_attn_ln), memory_config=memory)
            residual = hidden
            mlp_in = interleaved(norm(hidden, decoder.weights.pre_ff_ln))
            hidden_1 = norm(sharded(decoder._dense_mlp(mlp_in, fold_activation=False)), decoder.weights.post_ff_ln_1)
            residual_interleaved = interleaved(residual)
            routing = decoder._router_weights(residual_interleaved)
            moe_in = interleaved(norm(residual, decoder.weights.pre_ff_ln_2))
            hidden_2 = norm(sharded(decoder._moe_decode(moe_in, routing)), decoder.weights.post_ff_ln_2)
            hidden = norm(ttnn.add(hidden_1, hidden_2, memory_config=memory), decoder.weights.post_ff_ln)
            hidden = ttnn.add(residual, hidden, memory_config=memory)
            return ttnn.mul(hidden, decoder.weights.layer_scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        try:
            output = candidate()
            correct, pcc = comp_pcc(reference_torch, _to_torch(mesh_device, output), 0.995)
            candidate()
            ttnn.synchronize_device(mesh_device)
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            traced_output = candidate()
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            samples = []
            for _ in range(21):
                start = time.perf_counter()
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                ttnn.synchronize_device(mesh_device)
                samples.append((time.perf_counter() - start) * 1000)
            ttnn.release_trace(mesh_device, trace_id)
            replay_correct, replay_pcc = comp_pcc(
                _to_torch(mesh_device, output), _to_torch(mesh_device, traced_output), 0.999
            )
            rows.append(
                {
                    "cores": cores,
                    "shard_shape": [32, 2816 // cores],
                    "norm_program": str(norm_program),
                    "whole_chain_trace_host_ms_median": statistics.median(samples),
                    "whole_chain_trace_host_ms_samples": samples,
                    "pcc": float(pcc),
                    "trace_replay_pcc": float(replay_pcc),
                    "status": "passed" if correct and replay_correct else "failed_pcc",
                }
            )
        except RuntimeError as error:
            rows.append({"cores": cores, "status": "runtime_error", "error": str(error)})
    artifact = {
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "logical_batch": 1,
        "rows": rows,
        "decision": "keep_interleaved" if not any(row["status"] == "passed" for row in rows) else "compare",
    }
    (ARTIFACT_DIR / f"residual_chain_layer{layer_idx}_{layer_type}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    assert len(rows) == 2


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_large_prefill_program_configs(mesh_device, device_params, layer_idx):
    if os.getenv("GEMMA4_OPTIMIZED_PREFILL_CONFIG") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_PREFILL_CONFIG=1 to run the large-prefill A/B")
    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    decoder = OptimizedDecoder.from_state_dict(
        _load_layer_state(layer_idx), hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device
    )
    prefill_args, _ = _make_perf_args(decoder, mesh_device, layer_type=layer_type, seq_len=1024, batch=1)

    def run(enabled):
        decoder.USE_LARGE_PREFILL_DENSE_CONFIGS = enabled
        start = time.perf_counter()
        output = decoder.prefill_forward(**prefill_args)
        ttnn.synchronize_device(mesh_device)
        return output, (time.perf_counter() - start) * 1000

    run(False)
    run(True)
    baseline_samples = []
    candidate_samples = []
    baseline_output = candidate_output = None
    for _ in range(7):
        baseline_output, baseline_ms = run(False)
        candidate_output, candidate_ms = run(True)
        baseline_samples.append(baseline_ms)
        candidate_samples.append(candidate_ms)
    passed, pcc = comp_pcc(_to_torch(mesh_device, baseline_output), _to_torch(mesh_device, candidate_output), 0.999)
    baseline_median = statistics.median(baseline_samples)
    candidate_median = statistics.median(candidate_samples)
    artifact = {
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "logical_batch": 1,
        "sequence_length": 1024,
        "gate_up_config": "grid=6x8, M=4, Kblock=11, per_core_N=11",
        "down_config": "grid=8x8, M=4, Kblock=6, per_core_N=11",
        "pcc": float(pcc),
        "baseline_host_ms_median": baseline_median,
        "baseline_host_ms_samples": baseline_samples,
        "candidate_host_ms_median": candidate_median,
        "candidate_host_ms_samples": candidate_samples,
        "speedup": baseline_median / candidate_median,
        "decision": "keep" if passed and candidate_median < baseline_median else "reject",
    }
    (ARTIFACT_DIR / f"large_prefill_layer{layer_idx}_{layer_type}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    assert passed, pcc


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_sparse_prefill_geometry(mesh_device, device_params, layer_idx):
    if os.getenv("GEMMA4_OPTIMIZED_PREFILL_SPARSE") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_PREFILL_SPARSE=1 to run sparse prefill geometry")
    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    decoder = OptimizedDecoder.from_state_dict(
        _load_layer_state(layer_idx), hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device
    )
    prefill_args, _ = _make_perf_args(decoder, mesh_device, layer_type=layer_type, seq_len=1024, batch=1)
    candidates = [
        ("u4_d8_k1_1", 4, 8, 1, 1),
        ("u2_d4_k4_2", 2, 4, 4, 2),
        ("u2_d4_k11_11", 2, 4, 11, 11),
        ("u4_d8_k8_11", 4, 8, 8, 11),
        ("u4_d8_k22_22", 4, 8, 22, 22),
    ]
    rows = []
    reference = None

    def output_subblock_w(per_core_n):
        width = min(4, per_core_n)
        while per_core_n % width:
            width -= 1
        return width

    for name, up_cores, down_cores, up_block, down_block in candidates:
        decoder.PREFILL_EXPERT_UP_GATE_CORES = up_cores
        decoder.PREFILL_EXPERT_DOWN_CORES = down_cores
        decoder.PREFILL_EXPERT_UP_GATE_IN0_BLOCK_W = up_block
        decoder.PREFILL_EXPERT_DOWN_IN0_BLOCK_W = down_block
        decoder.prefill_forward(**prefill_args)
        samples = []
        output = None
        for _ in range(5):
            start = time.perf_counter()
            output = decoder.prefill_forward(**prefill_args)
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter() - start) * 1000)
        output_torch = _to_torch(mesh_device, output)
        if reference is None:
            reference = output_torch
        passed, pcc = comp_pcc(reference, output_torch, 0.999)
        rows.append(
            {
                "candidate": name,
                "up_gate_cores": up_cores,
                "down_cores": down_cores,
                "up_gate_in0_block_w": up_block,
                "down_in0_block_w": down_block,
                "up_gate_per_core_n": 44 // up_cores,
                "down_per_core_n": 88 // down_cores,
                "out_subblock_h": 1,
                "up_gate_out_subblock_w": output_subblock_w(44 // up_cores),
                "down_out_subblock_w": output_subblock_w(88 // down_cores),
                "prefill_host_ms_samples": samples,
                "prefill_host_ms_median": statistics.median(samples),
                "pcc_vs_control": float(pcc),
                "status": "passed" if passed else "failed_pcc",
            }
        )
    passed_rows = [row for row in rows if row["status"] == "passed"]
    winner = min(passed_rows, key=lambda row: row["prefill_host_ms_median"])["candidate"]
    artifact = {
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "logical_batch": 1,
        "sequence_length": 1024,
        "weight_dtype": "BFLOAT8_B",
        "math_fidelity": "LoFi",
        "rows": rows,
        "winner": winner,
    }
    (ARTIFACT_DIR / f"prefill_sparse_geometry_layer{layer_idx}_{layer_type}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    selected = (
        f"u{OptimizedDecoder.PREFILL_EXPERT_UP_GATE_CORES}_"
        f"d{OptimizedDecoder.PREFILL_EXPERT_DOWN_CORES}_"
        f"k{OptimizedDecoder.PREFILL_EXPERT_UP_GATE_IN0_BLOCK_W}_"
        f"{OptimizedDecoder.PREFILL_EXPERT_DOWN_IN0_BLOCK_W}"
    )
    assert winner == selected, (winner, selected)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_non_aligned_prefill(mesh_device, device_params, layer_idx, monkeypatch):
    monkeypatch.setenv("GEMMA4_BOUNDARY_LENGTHS", "31,33,1025")
    with _optimized_acceptance_oracle():
        functional_tests.test_paged_prefill_logical_boundary_lengths(mesh_device, device_params, layer_idx)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_bounded_modulo_cache_integrity(mesh_device, device_params):
    with _optimized_acceptance_oracle():
        functional_tests.test_bounded_modulo_prefill_tail_cache_integrity(mesh_device, device_params)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_advertised_context_decode(mesh_device, device_params, layer_idx):
    if os.getenv("GEMMA4_OPTIMIZED_CONTEXT") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_CONTEXT=1 to run advertised-context decode")
    old = os.environ.get("GEMMA4_FUNCTIONAL_DECODER_CONTEXT")
    os.environ["GEMMA4_FUNCTIONAL_DECODER_CONTEXT"] = "1"
    try:
        with _optimized_acceptance_oracle():
            functional_tests.test_advertised_context_traced_decode(mesh_device, device_params, layer_idx)
    finally:
        if old is None:
            os.environ.pop("GEMMA4_FUNCTIONAL_DECODER_CONTEXT", None)
        else:
            os.environ["GEMMA4_FUNCTIONAL_DECODER_CONTEXT"] = old


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_repeated_trace_stress(mesh_device, device_params, layer_idx):
    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    decoder = OptimizedDecoder.from_state_dict(
        _load_layer_state(layer_idx), hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device
    )
    _, decode_args = _make_perf_args(decoder, mesh_device, layer_type=layer_type, seq_len=1024, batch=1)
    eager, traced, samples = _measure_decode_trace(decoder, mesh_device, decode_args, repeats=101)
    passed, pcc = comp_pcc(_to_torch(mesh_device, eager), _to_torch(mesh_device, traced), 0.9999)
    artifact = {
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "logical_batch": 1,
        "sequence_length": 1024,
        "trace_replays": 101,
        "eager_vs_replay_pcc": float(pcc),
        "trace_host_ms_median": statistics.median(samples),
        "trace_host_ms_samples": samples,
    }
    (ARTIFACT_DIR / f"stress_trace_layer{layer_idx}_{layer_type}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    assert passed, pcc


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical", [(0, True), (5, False)], ids=["sliding_attention", "full_attention"]
)
def test_optimized_decoder_perf_profile(mesh_device, device_params, layer_idx, shared_physical):
    if os.getenv("GEMMA4_OPTIMIZED_PROFILE") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_PROFILE=1 to run the profiler harness")
    old = os.environ.get("GEMMA4_FUNCTIONAL_DECODER_PERF")
    os.environ["GEMMA4_FUNCTIONAL_DECODER_PERF"] = "1"
    try:
        with _optimized_acceptance_oracle():
            functional_tests.test_functional_decoder_perf_profile(
                mesh_device, device_params, layer_idx, shared_physical, 1
            )
    finally:
        if old is None:
            os.environ.pop("GEMMA4_FUNCTIONAL_DECODER_PERF", None)
        else:
            os.environ["GEMMA4_FUNCTIONAL_DECODER_PERF"] = old


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize(
    "candidate",
    [
        "bfp4_experts_isolated",
        "bfp8_experts_hifi4",
        "bfp8_experts_lofi",
        "bfp4_attention_only",
        "bfp8_attention_only",
        "bfp4_dense_gate_up",
        "bfp4_dense_all",
        "bfp4_dense_gate_up_packed",
        "bfp8_projection_hifi2",
    ],
)
def test_optimized_precision_candidate(mesh_device, device_params, layer_idx, candidate):
    if os.getenv("GEMMA4_OPTIMIZED_CANDIDATES") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_CANDIDATES=1 to run the real-weight policy sweep")
    cfg = _load_text_config()
    state = _load_layer_state(layer_idx)
    layer_type = cfg.layer_types[layer_idx]
    fused = FusedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    optimized = OptimizedDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        candidate=candidate,
    )
    fused_args = _make_perf_args(fused, mesh_device, layer_type=layer_type, seq_len=1024, batch=1)
    optimized_args = _make_perf_args(optimized, mesh_device, layer_type=layer_type, seq_len=1024, batch=1)
    fused_output = fused.decode_forward(**fused_args[1])
    optimized_output = optimized.decode_forward(**optimized_args[1])
    passed, pcc = comp_pcc(_to_torch(mesh_device, fused_output), _to_torch(mesh_device, optimized_output), 0.99)
    fused_perf = _measure_warmed(
        fused,
        mesh_device,
        prefill_args=fused_args[0],
        decode_args=fused_args[1],
        batch=1,
    )
    optimized_perf = _measure_warmed(
        optimized,
        mesh_device,
        prefill_args=optimized_args[0],
        decode_args=optimized_args[1],
        batch=1,
    )
    artifact = {
        "candidate": candidate,
        "policy": {name: str(value) for name, value in vars(POLICIES[candidate]).items()},
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "logical_batch": 1,
        "tile_padded_rows": 32,
        "fused_vs_candidate_decode_pcc": float(pcc),
        "fused": fused_perf,
        "candidate_result": optimized_perf,
        "decode_speedup_vs_fused": fused_perf["decode_trace_host_ms_median"]
        / optimized_perf["decode_trace_host_ms_median"],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"precision_{candidate}_layer{layer_idx}_{layer_type}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    assert passed, pcc


def _measure_decode_trace(decoder, mesh_device, decode_args, repeats=21):
    eager = decoder.decode_forward(**decode_args)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = decoder.decode_forward(**decode_args)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
        samples.append((time.perf_counter() - start) * 1000)
    ttnn.release_trace(mesh_device, trace_id)
    return eager, traced, samples


def _geometry_descriptor(name, policy):
    return {
        "candidate": name,
        "up_gate_cores": policy.expert_up_gate_cores,
        "down_cores": policy.expert_down_cores,
        "up_gate_in0_block_w": policy.expert_up_gate_in0_block_w,
        "down_in0_block_w": policy.expert_down_in0_block_w,
        "up_gate_per_core_n": 44 // policy.expert_up_gate_cores,
        "down_per_core_n": 88 // policy.expert_down_cores,
    }


@pytest.mark.parametrize("precision", ["bfp4", "bfp8"])
def test_optimized_single_core_sparse_geometry_is_host_rejected(precision):
    names = sorted(name for name in POLICIES if name.startswith(f"{precision}_geo_"))
    rejected = [name for name in names if sparse_geometry_host_rejection(POLICIES[name])]
    assert rejected == [
        f"{precision}_geo_u1_d2_k1_1",
        f"{precision}_geo_u1_d2_k22_22",
    ]
    assert all(POLICIES[name].expert_up_gate_cores in (2, 4) for name in set(names) - set(rejected))
    with pytest.raises(ValueError, match=SPARSE_SINGLE_CORE_MCAST_BLOCKER):
        _sparse_program_config(
            1,
            2 * 1408,
            in0_block_w=1,
            num_cores=1,
            projection="expert_up_gate",
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("precision", ["bfp4", "bfp8"])
def test_optimized_sparse_geometry_sweep(mesh_device, device_params, layer_idx, precision):
    if os.getenv("GEMMA4_OPTIMIZED_GEOMETRY") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_GEOMETRY=1 to run sparse geometry sweep")
    cfg = _load_text_config()
    state = _load_layer_state(layer_idx)
    layer_type = cfg.layer_types[layer_idx]
    names = sorted(name for name in POLICIES if name.startswith(f"{precision}_geo_"))
    rows = []
    runnable_names = []
    for name in names:
        policy = POLICIES[name]
        if rejection := sparse_geometry_host_rejection(policy):
            rows.append(
                {
                    **_geometry_descriptor(name, policy),
                    "status": "host_rejected",
                    "blocker": SPARSE_SINGLE_CORE_MCAST_BLOCKER,
                    "reason": rejection,
                }
            )
        else:
            runnable_names.append(name)
    assert runnable_names, f"no safe {precision} sparse geometry candidates"
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        candidate=runnable_names[0],
    )
    _, decode_args = _make_perf_args(decoder, mesh_device, layer_type=layer_type, seq_len=1024, batch=1)
    reference = None
    for name in runnable_names:
        policy = POLICIES[name]
        decoder._apply_policy(name, policy)
        try:
            eager, traced, samples = _measure_decode_trace(decoder, mesh_device, decode_args)
            eager_torch = _to_torch(mesh_device, eager)
            trace_torch = _to_torch(mesh_device, traced)
            if reference is None:
                reference = eager_torch
            passed, pcc = comp_pcc(reference, eager_torch, 0.99)
            replay_passed, replay_pcc = comp_pcc(eager_torch, trace_torch, 0.999)
            rows.append(
                {
                    **_geometry_descriptor(name, policy),
                    "trace_host_ms_samples": samples,
                    "trace_host_ms_median": statistics.median(samples),
                    "correctness_pcc": float(pcc),
                    "trace_replay_pcc": float(replay_pcc),
                    "status": "passed" if passed and replay_passed else "failed_pcc",
                }
            )
        except RuntimeError as error:
            rows.append({**_geometry_descriptor(name, policy), "status": "runtime_error", "error": str(error)})
    passed_rows = [row for row in rows if row["status"] == "passed"]
    artifact = {
        "precision": precision,
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "logical_batch": 1,
        "tile_padded_rows": 32,
        "rows": rows,
        "winner": min(passed_rows, key=lambda row: row["trace_host_ms_median"])["candidate"],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"geometry_{precision}_layer{layer_idx}_{layer_type}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    assert len(passed_rows) >= 2, artifact


def test_optimized_default_is_not_functional_fallback():
    assert OptimizedDecoder.DEFAULT_CANDIDATE in POLICIES
    assert OptimizedDecoder.decode_forward is not FusedDecoder.decode_forward
    assert OptimizedDecoder.decode_forward is not functional_tests.FunctionalDecoder.decode_forward
    source = inspect.getsource(OptimizedDecoder.decode_forward)
    assert "self._residual_norm" in source
    assert "self._moe_decode" in source
