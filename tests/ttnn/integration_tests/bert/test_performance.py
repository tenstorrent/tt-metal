# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from loguru import logger
import torch
import transformers


import ttnn

from models.experimental.functional_bert.tt import ttnn_functional_bert
from models.experimental.functional_bert.tt import ttnn_optimized_functional_bert

from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import (
    skip_for_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report


def get_expected_times(functional_bert):
    return {
        ttnn_functional_bert: (12, 17),
        ttnn_optimized_functional_bert: (12, 0.08),
    }[functional_bert]


@skip_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("functional_bert", [ttnn_functional_bert, ttnn_optimized_functional_bert])
def test_performance(device, use_program_cache, model_name, batch_size, sequence_size, functional_bert):
    disable_persistent_kernel_cache()

    config = transformers.BertConfig.from_pretrained(model_name)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.zeros(1, sequence_size) if functional_bert == ttnn_optimized_functional_bert else None

    if functional_bert == ttnn_functional_bert:
        tt_model_name = f"ttnn_{model_name}"
    elif functional_bert == ttnn_optimized_functional_bert:
        tt_model_name = f"ttnn_optimized_{model_name}"
    else:
        raise ValueError(f"Unknown functional_bert: {functional_bert}")

    parameters = preprocess_model_parameters(
        tt_model_name,
        initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        custom_preprocessor=functional_bert.custom_preprocessor,
        device=device,
    )

    durations = []
    for _ in range(2):
        ttnn_bert_inputs = functional_bert.preprocess_inputs(
            input_ids,
            torch_token_type_ids,
            torch_attention_mask,
            device=device,
        )

        start = time.time()
        tt_output = functional_bert.bert_for_question_answering(
            config,
            *ttnn_bert_inputs,
            parameters=parameters,
        )
        tt_output = ttnn.from_device(tt_output)
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times(functional_bert)
    prep_perf_report(
        model_name=tt_model_name,
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
