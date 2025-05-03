# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

from loguru import logger
import pytest
from transformers import BloomConfig, BloomForCausalLM, BloomForQuestionAnswering, BloomTokenizerFast

from models.demos.grayskull.functional_bloom.tt import ttnn_functional_bloom
from models.demos.grayskull.functional_bloom.tt import ttnn_optimized_functional_bloom
from models.utility_functions import (
    is_wormhole_b0,
    is_blackhole,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters


def get_expected_times_qa(functional_bloom):
    return {
        ttnn_functional_bloom: (15.0, 9.2),
        ttnn_optimized_functional_bloom: (12, 0.85),
    }[functional_bloom]


def get_expected_times_causal_lm(functional_bloom):
    return {
        ttnn_functional_bloom: (16.0, 10.0),
        ttnn_optimized_functional_bloom: (9.0, 7.8),
    }[functional_bloom]


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("functional_bloom", [ttnn_functional_bloom, ttnn_optimized_functional_bloom])
def test_performance_of_bloom_for_question_answering(
    device, use_program_cache, functional_bloom, batch_size=8, max_length=384
):
    disable_persistent_kernel_cache()

    model_name = "bigscience/bloom-560m"
    config = BloomConfig.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    num_heads = config.n_head

    question = "What is my name?"
    context = "My name is John."
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")

    if functional_bloom == ttnn_functional_bloom:
        tt_model_name = f"ttnn_{model_name}"
    elif functional_bloom == ttnn_optimized_functional_bloom:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_bloom: {functional_bloom}")

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: BloomForQuestionAnswering.from_pretrained(model_name).eval(),
        device=device,
        custom_preprocessor=functional_bloom.custom_preprocessor,
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    num_tokens = input_ids.shape[-1]
    input_ids = input_ids.expand((batch_size, num_tokens))
    attention_mask = attention_mask.expand((batch_size, num_tokens))

    input_ids, alibi, causal_mask = functional_bloom.preprocess_inputs(
        input_ids=input_ids, device=device, num_heads=num_heads, attention_mask=attention_mask, max_length=max_length
    )

    # TODO: don't modify the config globally. Pass it into the functions instead
    ttnn_optimized_functional_bloom.ASSUME_FUSED_SOFTMAX = True

    durations = []
    for _ in range(2):
        start = time.time()
        tt_output = functional_bloom.bloom_for_question_answering(
            config, input_ids, alibi, causal_mask, parameters=parameters
        )
        tt_output = ttnn.from_device(tt_output)
        end = time.time()

        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times_qa(functional_bloom)
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

    # TODO: don't modify the config globally. Pass it into the functions instead
    ttnn_optimized_functional_bloom.ASSUME_FUSED_SOFTMAX = False


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("functional_bloom", [ttnn_functional_bloom, ttnn_optimized_functional_bloom])
def test_performance_of_causal_lm(device, use_program_cache, functional_bloom, batch_size=8, max_length=128):
    disable_persistent_kernel_cache()

    model_name = "bigscience/bloom-560m"
    config = BloomConfig.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    num_heads = config.n_head

    context = "Hello, my dog is cute"
    inputs = tokenizer.encode_plus(context, return_tensors="pt")

    if functional_bloom == ttnn_functional_bloom:
        tt_model_name = f"ttnn_{model_name}"
    elif functional_bloom == ttnn_optimized_functional_bloom:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_bloom: {functional_bloom}")

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: BloomForCausalLM.from_pretrained(model_name).eval(),
        device=device,
        convert_to_ttnn=lambda model, name: name != "lm_head",
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    num_tokens = input_ids.shape[-1]
    input_ids = input_ids.expand((batch_size, num_tokens))
    attention_mask = attention_mask.expand((batch_size, num_tokens))

    input_ids, alibi, causal_mask = functional_bloom.preprocess_inputs(
        input_ids=input_ids, device=device, num_heads=num_heads, attention_mask=attention_mask, max_length=max_length
    )

    # TODO: don't modify the config globally. Pass it into the functions instead
    ttnn_optimized_functional_bloom.ASSUME_FUSED_SOFTMAX = True

    durations = []
    for _ in range(2):
        start = time.time()
        tt_output = functional_bloom.bloom_for_causal_lm(config, input_ids, alibi, causal_mask, parameters=parameters)
        tt_output = ttnn.from_torch(tt_output, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        end = time.time()

        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times_causal_lm(functional_bloom)
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

    # TODO: don't modify the config globally. Pass it into the functions instead
    ttnn_optimized_functional_bloom.ASSUME_FUSED_SOFTMAX = False
