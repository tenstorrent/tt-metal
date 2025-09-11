# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger

from models.perf.device_perf_utils import check_device_perf, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import is_wormhole_b0, run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, expected_perf, test, pcc_file, function_name",
    [
        # [256, 14000, "sentence_bert_tg", "test_ttnn_sentencebert_model", "test_ttnn_sentence_bert_model"],
        [256, 577315, "sentence_bert_tg", "test_ttnn_sentencebert_attention", "test_ttnn_sentence_bert_attention"],
        [256, 617787, "sentence_bert_tg", "test_ttnn_sentencebert_embeddings", "test_ttnn_sentence_bert_Embeddings"],
        # [256, 14867, "sentence_bert_tg", "test_ttnn_sentencebert_encoder", "test_ttnn_sentence_bert_encoder"],
        [
            256,
            813470,
            "sentence_bert_tg",
            "test_ttnn_sentencebert_intermediate",
            "test_ttnn_sentence_bert_intermediate",
        ],
        [256, 178428, "sentence_bert_tg", "test_ttnn_sentencebert_layer", "test_ttnn_sentence_bert_layer"],
        [256, 4843073, "sentence_bert_tg", "test_ttnn_sentencebert_pooler", "test_ttnn_sentence_bert_pooler"],
        [
            256,
            761329,
            "sentence_bert_tg",
            "test_ttnn_sentencebert_self_attention",
            "test_ttnn_sentence_bert_self_attention",
        ],
        [256, 2035931, "sentence_bert_tg", "test_ttnn_sentencebert_self_output", "test_ttnn_sentence_bert_self_output"],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_sentence_bert_tg(batch_size, expected_perf, test, pcc_file, function_name):
    """Test device performance for SentenceBERT TG model on Galaxy systems."""
    subdir = "ttnn_sentence_bert_model_tg"
    num_iterations = 1
    margin = 0.05
    expected_perf = expected_perf if is_wormhole_b0() else 0
    expected_inference_time = 1 / (expected_perf * (1 - margin))

    command = f"pytest models/demos/sentence_bert/tests/pcc/{pcc_file}.py::{function_name}"
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
        model_name=f"perf_{pcc_file}_{today}",
        batch_size=batch_size,
        inference_and_compile_time=inference_time,  # Use inference time as placeholder
        inference_time=inference_time,
        expected_compile_time=60,  # Placeholder value
        expected_inference_time=expected_inference_time,
        comments=test.replace("/", "_"),
        inference_time_cpu=None,
    )
