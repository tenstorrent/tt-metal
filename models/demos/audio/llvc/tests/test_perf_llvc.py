# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LLVC Performance Test — generates perf report CSV using prep_perf_report().

Follows the standard tt-metal perf test pattern (see test_perf_distilbert.py).
Runs compile+inference timing on TT device and outputs the perf CSV.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.audio.llvc.reference.model import Net, get_default_config
from models.demos.audio.llvc.tt.ttnn_functional_llvc import preprocess_model_parameters, ttnn_llvc_forward
from models.perf.perf_utils import prep_perf_report

SAMPLE_RATE = 16000
LLVC_L1_SMALL_SIZE = 16384


def _has_tt_device():
    try:
        return ttnn.GetNumAvailableDevices() > 0
    except Exception:
        return os.path.exists("/dev/tenstorrent")


def _create_reference_model():
    config = get_default_config()
    model = Net(config)
    model.eval()
    return config, model


@pytest.mark.skipif(not _has_tt_device(), reason="No TT device available")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": LLVC_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, audio_duration_s, expected_inference_time, expected_compile_time",
    ([1, 0.1, 0.050, 60.0],),
)
def test_perf_llvc(
    device,
    batch_size,
    audio_duration_s,
    expected_inference_time,
    expected_compile_time,
):
    """LLVC end-to-end performance test with perf report generation.

    Measures compile time (first run) and inference time (second run),
    then generates a CSV perf report using prep_perf_report().
    """
    torch.manual_seed(42)

    config, reference_model = _create_reference_model()
    parameters = preprocess_model_parameters(reference_model, device=device)
    ttnn_config = {
        "L": config["L"],
        "enc_dim": config["enc_dim"],
        "num_enc_layers": config["num_enc_layers"],
        "dec_dim": config["dec_dim"],
        "num_dec_layers": config["num_dec_layers"],
        "dec_buf_len": config["dec_buf_len"],
        "dec_chunk_size": config["dec_chunk_size"],
        "out_buf_len": config["out_buf_len"],
    }
    L = config["L"]

    T = int(audio_duration_s * SAMPLE_RATE)
    T = (T // L) * L
    x = torch.randn(batch_size, 1, T)

    # CPU reference timing
    with torch.no_grad():
        cpu_start = time.time()
        _ = reference_model(x, pad=True)
        cpu_elapsed = time.time() - cpu_start

    # Run twice: first run includes compile, second is pure inference
    durations = []
    for run_idx in range(2):
        x_tt = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        start = time.time()
        tt_out = ttnn_llvc_forward(
            x_tt,
            parameters=parameters,
            config=ttnn_config,
            device=device,
            pad=True,
        )
        # Force sync by reading output
        _ = ttnn.to_torch(tt_out)
        end = time.time()
        durations.append(end - start)

    inference_and_compile_time, inference_time, *_ = durations
    compile_time = inference_and_compile_time - inference_time

    num_tokens = T // L
    tokens_per_sec = num_tokens / inference_time
    rtf = inference_time / audio_duration_s

    # Generate perf report CSV (standard tt-metal format)
    prep_perf_report(
        model_name="ttnn_llvc",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=f"audio_{audio_duration_s}s_N300",
        inference_time_cpu=cpu_elapsed,
    )

    logger.info("LLVC Performance Report:")
    logger.info(f"  Compile time: {compile_time:.2f}s")
    logger.info(f"  Inference time: {inference_time * 1000:.1f}ms")
    logger.info(f"  Tokens/sec: {tokens_per_sec:.0f}")
    logger.info(f"  RTF: {rtf:.4f}")
    logger.info(f"  CPU inference time: {cpu_elapsed * 1000:.1f}ms")
    logger.info(f"  Device throughput: {batch_size / inference_time:.1f} samples/s")
    logger.info(f"  CPU throughput: {batch_size / cpu_elapsed:.1f} samples/s")

    # Assertions (Stage 1 targets)
    assert tokens_per_sec >= 50, f"Tokens/sec {tokens_per_sec:.1f} below Stage 1 minimum 50"
    assert rtf < 0.3, f"RTF {rtf:.4f} exceeds Stage 1 target 0.3"
    assert (
        inference_time < expected_inference_time
    ), f"Inference time {inference_time:.4f}s exceeds expected {expected_inference_time:.4f}s"

    logger.info("LLVC perf test PASSED")
