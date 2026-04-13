# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Compare conditional-generation inference perf: legacy KV prefill (token-by-token) vs batched prefill.

Runs four repeats per mode, then logs one summary line per mode with mean metrics.
"""

import math
import os
import statistics

import pytest
from loguru import logger

import ttnn
from models.demos.audio.whisper.demo.demo import (
    GenerationParams,
    format_conditional_generation_inference_perf_metrics,
    run_demo_whisper_for_conditional_generation_inference,
)
from models.demos.audio.whisper.tt.ttnn_optimized_functional_whisper import (
    WHISPER_L1_SMALL_SIZE,
    WHISPER_TRACE_REGION_SIZE,
)

# Match demo.py device selection for conditional generation
available_devices = len(ttnn.get_device_ids()) if ttnn.get_device_ids() else 1

NUM_BENCHMARK_REPEATS = 4


@pytest.mark.parametrize(
    "input_path",
    (["models/demos/audio/whisper/demo/dataset/conditional_generation"],),
)
@pytest.mark.parametrize("model_repo", ("distil-whisper/distil-large-v3",))
@pytest.mark.parametrize("batch_size_per_device", [2])
@pytest.mark.parametrize("num_inputs", [2])
@pytest.mark.parametrize("prompt", ["Glossary: medal podium, levee."])
@pytest.mark.parametrize(
    "mesh_device",
    [available_devices]
    if os.getenv("CI") != "true"
    else ([1, available_devices] if available_devices != 1 else [available_devices]),
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": WHISPER_L1_SMALL_SIZE, "trace_region_size": WHISPER_TRACE_REGION_SIZE, "num_command_queues": 2}],
    indirect=True,
)
def test_conditional_generation_inference_prefill_modes_benchmark(
    input_path,
    mesh_device,
    model_repo,
    batch_size_per_device,
    num_inputs,
    prompt,
):
    """
    Mean over ``NUM_BENCHMARK_REPEATS`` runs per mode: batched vs incremental KV prefill.

    ``num_inputs`` must be >= 2 so averaged TTFT excludes one warmup batch (see demo loop).
    """
    generation_params = GenerationParams(
        temperatures=0.0,
        compression_ratio_threshold=None,
        logprob_threshold=None,
        no_speech_threshold=None,
        return_timestamps=False,
    )

    common_kwargs = dict(
        input_path=input_path[0],
        mesh_device=mesh_device,
        num_inputs=num_inputs,
        model_repo=model_repo,
        generation_params=generation_params,
        language="English",
        task="transcribe",
        prompt=prompt,
        batch_size_per_device=batch_size_per_device,
        stream=False,
        run_both_batch_sizes=False,
    )

    batched_runs = []
    incremental_runs = []

    for _ in range(NUM_BENCHMARK_REPEATS):
        logger.info("Running batched prefill benchmark")
        ttft_batched, decode_tpu_batched = run_demo_whisper_for_conditional_generation_inference(
            **common_kwargs,
            use_batched_decoder_prefill=True,
        )
        batched_runs.append(
            format_conditional_generation_inference_perf_metrics(
                mesh_device, batch_size_per_device, ttft_batched, decode_tpu_batched
            )
        )
        logger.info("Finished batched prefill benchmark")
        logger.info("Running incremental prefill benchmark")
        ttft_incr, decode_tpu_incr = run_demo_whisper_for_conditional_generation_inference(
            **common_kwargs,
            use_batched_decoder_prefill=False,
        )
        incremental_runs.append(
            format_conditional_generation_inference_perf_metrics(
                mesh_device, batch_size_per_device, ttft_incr, decode_tpu_incr
            )
        )
        logger.info("Finished incremental prefill benchmark")
    keys = ("prefill_time_to_token", "decode_t/s", "decode_t/s/u")
    for m in batched_runs + incremental_runs:
        for k in keys:
            assert math.isfinite(m[k])

    def _mean(runs, key):
        return statistics.mean(m[key] for m in runs)

    mean_batched = {k: _mean(batched_runs, k) for k in keys}
    mean_incremental = {k: _mean(incremental_runs, k) for k in keys}

    prompt_label = "no_prompt" if prompt is None else "with_prompt"
    logger.info(
        "[prefill benchmark | {} | mean of {} runs]\n"
        "  [batched KV prefill]     prefill_time_to_token={:.6f}s  decode_t/s={:.3f}  decode_t/s/u={:.3f}\n"
        "  [incremental KV prefill] prefill_time_to_token={:.6f}s  decode_t/s={:.3f}  decode_t/s/u={:.3f}",
        prompt_label,
        NUM_BENCHMARK_REPEATS,
        mean_batched["prefill_time_to_token"],
        mean_batched["decode_t/s"],
        mean_batched["decode_t/s/u"],
        mean_incremental["prefill_time_to_token"],
        mean_incremental["decode_t/s"],
        mean_incremental["decode_t/s/u"],
    )
