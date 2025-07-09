# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
import transformers
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.sentence_bert.reference.sentence_bert import BertModel, custom_extended_mask
from models.demos.sentence_bert.ttnn.common import custom_preprocessor, preprocess_inputs
from models.demos.sentence_bert.ttnn.ttnn_sentence_bert_model import TtnnSentenceBertModel
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import enable_persistent_kernel_cache, is_wormhole_b0, run_for_wormhole_b0


def get_expected_times(name):
    base = {"sentence_bert": (21.79, 0.02)}
    return base[name]


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "inputs",
    [["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [8, 384]]],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_perf(device, inputs):
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).eval()
    config = transformers.BertConfig.from_pretrained(inputs[0])
    input_ids = torch.randint(low=0, high=config.vocab_size - 1, size=inputs[1], dtype=torch.int64)
    attention_mask = torch.ones(inputs[1][0], inputs[1][1])
    extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
    token_type_ids = torch.zeros(inputs[1], dtype=torch.int64)
    position_ids = torch.arange(0, inputs[1][1], dtype=torch.int64).unsqueeze(dim=0)
    reference_module = BertModel(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_module = TtnnSentenceBertModel(parameters=parameters, config=config)
    durations = []
    for i in range(2):
        (
            ttnn_input_ids,
            ttnn_token_type_ids,
            ttnn_position_ids,
            ttnn_extended_attention_mask,
            ttnn_attention_mask,
        ) = preprocess_inputs(input_ids, token_type_ids, position_ids, extended_mask, attention_mask, device)
        start = time.time()
        ttnn_model_output = ttnn_module(
            ttnn_input_ids,
            ttnn_extended_attention_mask,
            ttnn_attention_mask,
            ttnn_token_type_ids,
            ttnn_position_ids,
            device=device,
        )
        end = time.time()
        durations.append(end - start)
        for outputs in ttnn_model_output:
            ttnn.deallocate(outputs)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("sentence_bert")

    prep_perf_report(
        model_name="models/demos/sentence_bert",
        batch_size=inputs[1][0],
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Sentences per second: {1 / inference_time *inputs[1][0] }")
    assert (
        inference_time < expected_inference_time
    ), f"Expected inference time: {expected_inference_time} Actual inference time: {inference_time}"


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, expected_perf,test",
    [
        [8, 440.4, "sentence_bert"],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_sentence_bert(batch_size, expected_perf, test):
    subdir = "ttnn_sentence_bert_model"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 0

    command = f"pytest tests/ttnn/integration_tests/sentence_bert/test_ttnn_sentencebert_model.py::test_ttnn_sentence_bert_model"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_sentence_bert{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
