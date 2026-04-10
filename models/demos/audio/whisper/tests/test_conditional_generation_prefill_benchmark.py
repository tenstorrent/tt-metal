# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Compare conditional-generation inference perf: legacy KV prefill (token-by-token) vs batched prefill.

Uses ``run_demo_whisper_for_conditional_generation_inference`` with identical inputs; only
``use_batched_decoder_prefill`` differs. Logs the same metrics as ``demo.py`` perf checks
(``prefill_time_to_token``, ``decode_t/s``, ``decode_t/s/u``).
"""

import math
import os

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


@pytest.mark.parametrize(
    "input_path",
    (["models/demos/audio/whisper/demo/dataset/conditional_generation"],),
)
@pytest.mark.parametrize("model_repo", ("distil-whisper/distil-large-v3",))
@pytest.mark.parametrize("batch_size_per_device", [1])
@pytest.mark.parametrize("num_inputs", [2])
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
):
    """
    Run the same audio pipeline twice: stacked single-step KV prefill vs one batched prefill forward.

    ``num_inputs`` must be >= 2 so the demo's averaged TTFT excludes one warmup batch (see demo loop).
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
        prompt=None,
        batch_size_per_device=batch_size_per_device,
        stream=False,
        run_both_batch_sizes=False,
    )

    ttft_batched, decode_tpu_batched = run_demo_whisper_for_conditional_generation_inference(
        **common_kwargs,
        use_batched_decoder_prefill=True,
    )
    metrics_batched = format_conditional_generation_inference_perf_metrics(
        mesh_device, batch_size_per_device, ttft_batched, decode_tpu_batched
    )
    logger.info(
        "[batched KV prefill] prefill_time_to_token={:.6f}s decode_t/s={:.3f} decode_t/s/u={:.3f} — {}",
        metrics_batched["prefill_time_to_token"],
        metrics_batched["decode_t/s"],
        metrics_batched["decode_t/s/u"],
        metrics_batched,
    )

    ttft_incr, decode_tpu_incr = run_demo_whisper_for_conditional_generation_inference(
        **common_kwargs,
        use_batched_decoder_prefill=False,
    )
    metrics_incremental = format_conditional_generation_inference_perf_metrics(
        mesh_device, batch_size_per_device, ttft_incr, decode_tpu_incr
    )
    logger.info(
        "[incremental KV prefill] prefill_time_to_token={:.6f}s decode_t/s={:.3f} decode_t/s/u={:.3f} — {}",
        metrics_incremental["prefill_time_to_token"],
        metrics_incremental["decode_t/s"],
        metrics_incremental["decode_t/s/u"],
        metrics_incremental,
    )

    # Sanity: finite timings from both runs
    for m in (metrics_batched, metrics_incremental):
        assert math.isfinite(m["prefill_time_to_token"])
        assert math.isfinite(m["decode_t/s"])
        assert math.isfinite(m["decode_t/s/u"])
