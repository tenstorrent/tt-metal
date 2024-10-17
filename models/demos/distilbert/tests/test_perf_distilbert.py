# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
import time

from models.demos.distilbert.tt import ttnn_optimized_distilbert
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    profiler,
)
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.perf.perf_utils import prep_perf_report
from transformers import DistilBertForQuestionAnswering, AutoTokenizer
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import is_grayskull, is_wormhole_b0


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("model_name", ["distilbert-base-uncased-distilled-squad"])
@pytest.mark.parametrize(
    "batch_size, seq_len, expected_inference_time, expected_compile_time",
    ([8, 384, 15.00, 20.00],),
)
def test_performance_distilbert_for_qa(
    batch_size,
    model_name,
    seq_len,
    expected_inference_time,
    expected_compile_time,
    device,
):
    hugging_face_reference_model = DistilBertForQuestionAnswering.from_pretrained(model_name)
    hugging_face_reference_model.eval()

    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = hugging_face_reference_model.config

    disable_persistent_kernel_cache()

    cpu_key = "ref_key"

    context = batch_size * [
        "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."
    ]
    question = batch_size * ["What discipline did Winkelmann create?"]
    inputs = tokenizer(
        question,
        context,
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    profiler.start(f"preprocessing_parameter")
    parameters = preprocess_model_parameters(
        model_name=f"ttnn_{model_name}_optimized",
        initialize_model=lambda: DistilBertForQuestionAnswering.from_pretrained(model_name, torchscript=False).eval(),
        custom_preprocessor=ttnn_optimized_distilbert.custom_preprocessor,
        device=device,
    )
    profiler.end(f"preprocessing_parameter")

    mask_reshp = (batch_size, 1, 1, inputs["attention_mask"].shape[1])
    score_shape = (batch_size, 12, 384, 384)

    mask = (inputs["attention_mask"] == 0).view(mask_reshp).expand(score_shape)
    min_val = torch.zeros(score_shape)
    min_val_tensor = min_val.masked_fill(mask, torch.tensor(torch.finfo(torch.bfloat16).min))

    negative_val = torch.zeros(score_shape)
    negative_val_tensor = negative_val.masked_fill(mask, -1)
    min_val_tensor = ttnn.from_torch(min_val_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    negative_val_tensor = ttnn.from_torch(negative_val_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_out = hugging_face_reference_model(**inputs)
        profiler.end(cpu_key)

        durations = []
        for _ in range(2):
            position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
            profiler.start(f"preprocessing_input")
            input_ids, position_ids, attention_mask = ttnn_optimized_distilbert.preprocess_inputs(
                inputs["input_ids"],
                position_ids,
                inputs["attention_mask"],
                device=device,
            )
            profiler.end(f"preprocessing_input")

            start = time.time()
            tt_output = ttnn_optimized_distilbert.distilbert_for_question_answering(
                config,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                parameters=parameters,
                device=device,
                min_val_tensor=min_val_tensor,
                negative_val_tensor=negative_val_tensor,
            )
            tt_output = ttnn.from_device(tt_output)
            end = time.time()

            durations.append(end - start)
            enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    prep_perf_report(
        model_name=f"ttnn_{model_name}_optimized",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")

    assert (
        inference_time < expected_inference_time
    ), f"Expected inference time: {expected_inference_time} Actual inference time: {inference_time}"
    logger.info("Exit Distilbert perf test")


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test",
    [
        [8, "distilbert-base-uncased-distilled-squad"],
    ],
)
def test_distilbert_perf_device(batch_size, test, reset_seeds):
    subdir = "ttnn_distilbert"
    margin = 0.03
    num_iterations = 1
    if is_grayskull():
        expected_perf = 40.8772
    elif is_wormhole_b0():
        expected_perf = 19.80

    command = f"pytest tests/ttnn/integration_tests/distilbert/test_ttnn_distilbert.py::test_distilbert_for_question_answering[sequence_size=768-batch_size=8-model_name=distilbert-base-uncased-distilled-squad]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_distilbert{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
