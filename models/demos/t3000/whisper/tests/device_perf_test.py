# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger

from models.perf.device_perf_utils import check_device_perf, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import is_wormhole_b0, run_for_wormhole_b0


@pytest.mark.model_perf_t3000
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, expected_perf, test, pcc_file, function_name, test_filter",
    [
        # # Whisper Attention tests - using custom ids from test_whisper_modules.py
        # [1, 26, "whisper_attention_encoder", "test_whisper_modules", "test_whisper_attention", "encoder_attn"],
        # [1, 82, "whisper_attention_decoder_self", "test_whisper_modules", "test_whisper_attention", "decoder_self_attn"],
        # [1, 97, "whisper_attention_decoder_self_kv", "test_whisper_modules", "test_whisper_attention", "decoder_self_attn_kv_cache"],
        # [1, 361, "whisper_attention_decoder_cross", "test_whisper_modules", "test_whisper_attention", "decoder_cross_attn"],
        # # Whisper Encoder tests
        # [1, 19, "whisper_encoder_layer", "test_whisper_modules", "test_encoder_layer", "1500"],
        # [1, 0.6, "whisper_encoder", "test_whisper_modules", "test_encoder", "3000"],
        # # Whisper Decoder tests
        # [1, 18, "whisper_decoder_layer", "test_whisper_modules", "test_decoder_layer", "32-False"],
        # [1, 18, "whisper_decoder_layer_kv", "test_whisper_modules", "test_decoder_layer", "1-True"],
        # [1, 11, "whisper_decoder", "test_whisper_modules", "test_decoder", "32-False"],
        # [1, 11, "whisper_decoder_kv", "test_whisper_modules", "test_decoder", "1-True"],
        # # Full Whisper model tests
        [1, 0.56, "whisper_model", "test_whisper_modules", "test_ttnn_whisper", "32-False"],
        # [1, 0.56, "whisper_model_kv", "test_whisper_modules", "test_ttnn_whisper", "1-True"],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_whisper_t3000(batch_size, expected_perf, test, pcc_file, function_name, test_filter):
    """Test device performance for Whisper model on T3000 systems."""
    subdir = "ttnn_whisper_model_t3000"
    num_iterations = 1
    margin = 0.05
    expected_perf = expected_perf if is_wormhole_b0() else 0
    expected_inference_time = 1 / (expected_perf * (1 - margin))

    # Use -k filter to select specific parametrized test case
    command = f"pytest models/demos/t3000/whisper/tests/{pcc_file}.py::{function_name} -k '{test_filter}'"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    # Calculate performance metrics for the standard perf report
    inference_perf = post_processed_results.get(inference_time_key, 0)
    inference_time = 1 / inference_perf

    today = time.strftime("%Y_%m_%d")

    # Use standard perf report with custom filename
    prep_perf_report(
        model_name=f"perf_{test}_{today}",
        batch_size=batch_size,
        inference_and_compile_time=inference_time,  # Use inference time as placeholder
        inference_time=inference_time,
        expected_compile_time=60,  # Placeholder value
        expected_inference_time=expected_inference_time,
        comments=test.replace("/", "_"),
        inference_time_cpu=None,
    )
