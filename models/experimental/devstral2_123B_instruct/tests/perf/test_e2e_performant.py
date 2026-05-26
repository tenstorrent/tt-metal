# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end wall-clock performance test for Devstral-2-123B (Ministral3) on TTNN.

Mirrors the prefill+decode trace flow in ``demo/text_demo.py`` but isolates the
steady-state timing windows (compile passes excluded from inference numbers):

  1. Build TT model (untimed).
  2. Prefill: 1 compile pass -> capture trace -> N trace replays (timed).
  3. Decode:  1 compile pass -> capture trace -> M trace replays (timed).

Reports TTFT (single prefill trace replay), prefill latency, decode tokens/sec/user,
and a CSV/JSON report via the standard ``prep_perf_report`` + ``BenchmarkData`` plumbing.

Run::

    pytest models/experimental/devstral2_123B_instruct/tests/perf/test_e2e_performant.py -k L10
    pytest models/experimental/devstral2_123B_instruct/tests/perf/test_e2e_performant.py -k L88

Decode trace + 2CQ (CQ1 input H2D, CQ0 trace replay) is on by default
(``DEVSTRAL2_DECODE_TRACE_2CQ=1``; set ``0`` for single-CQ baseline).
"""

from __future__ import annotations

import time
from typing import Optional

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.devstral2_123B_instruct.demo.decode_trace_2cq import (
    DecodeTrace2CQ,
    decode_trace_2cq_enabled,
    num_command_queues_for_decode,
    signal_decode_step_done,
    stage_decode_inputs,
)
from models.experimental.devstral2_123B_instruct.demo.text_demo import _mesh_device_param
from models.experimental.devstral2_123B_instruct.tests._devstral_weights import (
    DEVSTRAL2_TEST_MAX_SEQ_LEN,
    model_prefill_weight_keys,
    require_hf_weights,
    require_text_config,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_123B_instruct.tt.tt_ministral3_model import (
    TtMinistral3ForCausalLM,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.perf_utils import prep_perf_report
from models.tt_transformers.tt.ccl import TT_CCL


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _replicated_input_ids(input_ids: torch.Tensor, mesh_device, *, on_device: bool):
    kwargs = {
        "dtype": ttnn.uint32,
        "layout": ttnn.ROW_MAJOR_LAYOUT,
        "mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device),
    }
    if on_device:
        kwargs["device"] = mesh_device
    return ttnn.from_torch(input_ids, **kwargs)


def _replicated_positions(positions: torch.Tensor, mesh_device, *, on_device: bool):
    kwargs = {
        "dtype": ttnn.int32,
        "layout": ttnn.ROW_MAJOR_LAYOUT,
        "mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device),
    }
    if on_device:
        kwargs["device"] = mesh_device
        kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
    return ttnn.from_torch(positions.reshape(-1).to(torch.int32), **kwargs)


def _build_model(mesh_device, num_layers: int, max_seq_len: int):
    text_cfg = require_text_config()
    full_layers = int(text_cfg.num_hidden_layers)
    n_layers = num_layers or full_layers
    assert n_layers <= full_layers, f"requested {n_layers} layers, model has {full_layers}"

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max_seq_len,
        max_batch_size=1,
    )

    base_keys = model_prefill_weight_keys(n_layers)
    want_lm_head = not args.tie_word_embeddings
    try:
        weight_keys = base_keys + (["lm_head.weight"] if want_lm_head else [])
        state_dict = require_hf_weights(weight_keys)
    except Exception:
        if want_lm_head:
            logger.warning("lm_head.weight unavailable on the Hub; falling back to tied embeddings.")
            state_dict = require_hf_weights(base_keys)
        else:
            raise

    tt_ccl = TT_CCL(mesh_device)
    return TtMinistral3ForCausalLM(args, mesh_device, state_dict, tt_ccl, num_layers=n_layers), args


def _run_devstral2_perf(
    mesh_device,
    num_layers: int,
    prompt_len: int,
    decode_iters: int,
    prefill_iters: int,
):
    padded_prompt_len = max(_round_up(prompt_len, 32), 32)
    max_seq_len = max(_round_up(padded_prompt_len + decode_iters + 1, 32), DEVSTRAL2_TEST_MAX_SEQ_LEN)

    t_build = time.time()
    model, args = _build_model(mesh_device, num_layers, max_seq_len)
    build_time = time.time() - t_build
    logger.info(f"Model built in {build_time:.1f}s (max_seq_len={max_seq_len})")

    # Deterministic synthetic prompt - we only care about throughput here.
    input_ids = torch.zeros((1, padded_prompt_len), dtype=torch.long)
    prefill_tokens_dev = _replicated_input_ids(input_ids, mesh_device, on_device=True)
    decode_tok_dev = _replicated_input_ids(torch.zeros((1, 1), dtype=torch.long), mesh_device, on_device=True)
    decode_pos_dev = _replicated_positions(
        torch.tensor([padded_prompt_len], dtype=torch.long), mesh_device, on_device=True
    )

    decode_2cq: Optional[DecodeTrace2CQ] = None
    if decode_trace_2cq_enabled():
        decode_2cq = DecodeTrace2CQ.create(mesh_device, decode_tok_dev, decode_pos_dev)
        logger.info("Decode trace 2CQ enabled for perf (CQ1=H2D, CQ0=trace replay).")

    prefill_trace_id = None
    decode_trace_id = None
    try:
        # Prefill compile pass (untimed).
        t = time.time()
        warm = model(prefill_tokens_dev, mode="prefill", start_pos=0)
        ttnn.synchronize_device(mesh_device)
        prefill_compile_time = time.time() - t
        warm.deallocate(True)

        # Capture prefill trace
        prefill_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        prefill_trace_logits = model(prefill_tokens_dev, mode="prefill", start_pos=0)
        ttnn.end_trace_capture(mesh_device, prefill_trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        # TTFT: one prefill trace replay (steady-state time to first-token logits).
        t = time.time()
        ttnn.execute_trace(mesh_device, prefill_trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ttft_s = time.time() - t

        # Additional prefill replays for throughput averaging.
        t = time.time()
        for _ in range(prefill_iters):
            ttnn.execute_trace(mesh_device, prefill_trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        prefill_replay_time = (time.time() - t) / prefill_iters
        prefill_trace_logits.deallocate(True)
        logger.info(
            f"Prefill (seq={padded_prompt_len}): compile={prefill_compile_time*1000:.0f}ms, "
            f"TTFT={ttft_s*1000:.1f}ms, replay={prefill_replay_time*1000:.1f}ms "
            f"({padded_prompt_len/prefill_replay_time:.1f} tok/s)"
        )

        # Decode compile pass (untimed).
        decode_token = 0
        decode_pos = padded_prompt_len
        t = time.time()
        if decode_2cq is not None:
            stage_decode_inputs(decode_2cq, mesh_device, decode_tok_dev, decode_pos_dev, decode_token, decode_pos)
        tt_out = model(decode_tok_dev, mode="decode", current_pos=decode_pos_dev)
        ttnn.synchronize_device(mesh_device)
        signal_decode_step_done(decode_2cq)
        decode_compile_time = time.time() - t
        tt_out.deallocate(True)

        # Capture decode trace bound to (decode_tok_dev, decode_pos_dev)
        if decode_2cq is not None:
            stage_decode_inputs(decode_2cq, mesh_device, decode_tok_dev, decode_pos_dev, decode_token, decode_pos)
        decode_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        decode_trace_logits = model(decode_tok_dev, mode="decode", current_pos=decode_pos_dev)
        ttnn.end_trace_capture(mesh_device, decode_trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        signal_decode_step_done(decode_2cq)

        # Time decode replays. With 2CQ, include per-step input H2D (matches text_demo / agent).
        t = time.time()
        for replay_idx in range(decode_iters):
            if decode_2cq is not None:
                replay_pos = decode_pos + replay_idx
                stage_decode_inputs(decode_2cq, mesh_device, decode_tok_dev, decode_pos_dev, decode_token, replay_pos)
            ttnn.execute_trace(mesh_device, decode_trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            if decode_2cq is not None:
                signal_decode_step_done(decode_2cq)
        decode_total_time = time.time() - t
        decode_replay_time = decode_total_time / decode_iters
        decode_tok_per_s = decode_iters / decode_total_time
        decode_trace_logits.deallocate(True)
        cq_note = " 2CQ" if decode_2cq is not None else ""
        logger.info(
            f"Decode{cq_note}: compile={decode_compile_time*1000:.0f}ms, "
            f"replay={decode_replay_time*1000:.2f}ms ({decode_tok_per_s:.2f} tok/s/user)"
        )

        return {
            "build_time_s": build_time,
            "prefill_compile_time_s": prefill_compile_time,
            "ttft_s": ttft_s,
            "ttft_ms": ttft_s * 1000,
            "prefill_replay_time_s": prefill_replay_time,
            "prefill_throughput_tok_per_s": padded_prompt_len / prefill_replay_time,
            "decode_compile_time_s": decode_compile_time,
            "decode_replay_time_s": decode_replay_time,
            "decode_throughput_tok_per_s_per_user": decode_tok_per_s,
            "padded_prompt_len": padded_prompt_len,
            "decode_iters": decode_iters,
            "decode_trace_2cq": float(decode_2cq is not None),
        }
    finally:
        if prefill_trace_id is not None:
            ttnn.release_trace(mesh_device, prefill_trace_id)
        if decode_trace_id is not None:
            ttnn.release_trace(mesh_device, decode_trace_id)


def _e2e_perf_device_params():
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 100_000_000,
        "num_command_queues": num_command_queues_for_decode(),
        "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    }


@pytest.mark.timeout(0)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "num_layers, prompt_len, decode_iters, prefill_iters, expected_compile_time, expected_inference_time",
    [
        (10, 128, 32, 3, 600.0, 0.05),
        (88, 128, 32, 3, 1800.0, 0.05),
    ],
    ids=["L10", "L88"],
)
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", [_e2e_perf_device_params()], indirect=True)
def test_devstral2_123B_instruct_e2e_performant(
    mesh_device,
    num_layers,
    prompt_len,
    decode_iters,
    prefill_iters,
    expected_compile_time,
    expected_inference_time,
):
    results = _run_devstral2_perf(
        mesh_device,
        num_layers=num_layers,
        prompt_len=prompt_len,
        decode_iters=decode_iters,
        prefill_iters=prefill_iters,
    )

    batch_size = 1
    model_name = f"devstral2_123B_instruct_L{num_layers}"
    comments = f"prefill{results['padded_prompt_len']}_decode{decode_iters}"

    # Treat the *decode trace replay* time as the steady-state inference time, and
    # everything else (build + both compile passes + prefill replays) as compile/warmup.
    inference_time = results["decode_replay_time_s"]
    inference_and_compile_time = (
        results["build_time_s"]
        + results["prefill_compile_time_s"]
        + results["prefill_replay_time_s"] * prefill_iters
        + results["decode_compile_time_s"]
        + inference_time * decode_iters
    )

    prep_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
    )

    # Per-metric measurements for the benchmark database.
    profiler = BenchmarkProfiler()
    profiler.start("run")
    profiler.end("run")
    step = "devstral2_e2e"
    profiler.start(step)
    profiler.end(step)
    benchmark_data = BenchmarkData()
    for k, v in results.items():
        benchmark_data.add_measurement(profiler, 0, step, k, float(v))
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="end_to_end_perf",
        ml_model_name=model_name,
        batch_size=batch_size,
    )

    logger.info(
        f"{model_name}: TTFT={results['ttft_ms']:.1f}ms, "
        f"decode {results['decode_throughput_tok_per_s_per_user']:.2f} tok/s/user "
        f"(prefill {results['prefill_throughput_tok_per_s']:.1f} tok/s on {results['padded_prompt_len']}-tok prompt)"
    )
