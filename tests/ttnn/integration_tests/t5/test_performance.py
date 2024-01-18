# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

from loguru import logger
import pytest
import torch
import transformers

from models.experimental.functional_t5.tt import ttnn_functional_t5
from models.experimental.functional_t5.tt import ttnn_optimized_functional_t5
from models.utility_functions import (
    torch_random,
    skip_for_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters


def get_expected_times(model_name, functional_t5):
    return {
        "t5-small": {
            ttnn_functional_t5: (12, 3),
            ttnn_optimized_functional_t5: (10, 1),
        },
        "google/flan-t5-small": {
            ttnn_functional_t5: (7, 3),
            ttnn_optimized_functional_t5: (5, 1),
        },
    }[model_name][functional_t5]


@skip_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [64])
@pytest.mark.parametrize("functional_t5", [ttnn_functional_t5, ttnn_optimized_functional_t5])
def test_t5_for_conditional_generation(device, use_program_cache, model_name, batch_size, sequence_size, functional_t5):
    disable_persistent_kernel_cache()

    config = transformers.T5Config.from_pretrained(model_name)

    torch_input_ids = torch_random((batch_size, sequence_size), 0, 10, dtype=torch.int64)
    torch_decoder_input_ids = torch_random((batch_size, sequence_size), 0, 10, dtype=torch.int64)

    if functional_t5 == ttnn_functional_t5:
        tt_model_name = f"ttnn_{model_name}"
    elif functional_t5 == ttnn_optimized_functional_t5:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_t5: {functional_t5}")

    parameters = preprocess_model_parameters(
        tt_model_name,
        initialize_model=lambda: transformers.T5ForConditionalGeneration.from_pretrained(model_name).eval(),
        custom_preprocessor=functional_t5.custom_preprocessor,
        device=device,
    )

    durations = []
    for _ in range(2):
        input_ids = ttnn.from_torch(torch_input_ids)
        input_ids = ttnn.to_device(input_ids, device)
        decoder_input_ids = ttnn.from_torch(torch_decoder_input_ids)
        decoder_input_ids = ttnn.to_device(decoder_input_ids, device)

        start = time.time()
        output, *_ = functional_t5.t5_for_conditional_generation(
            config,
            input_ids,
            decoder_input_ids,
            parameters=parameters,
        )
        output = ttnn.from_device(output)
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times(model_name, functional_t5)
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
    logger.info(f"Tokens per second: {1 / inference_time * batch_size}")
