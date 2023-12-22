# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from loguru import logger
import torch
import torch.nn.functional as F
import transformers


import ttnn

from models.experimental.functional_bert.tt.ttnn_functional_bert import (
    ttnn_bert_for_question_answering,
)

from models.experimental.functional_bert.tt.ttnn_optimized_functional_bert import (
    ttnn_optimized_bert_for_question_answering,
)

from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_bias,
    preprocess_linear_weight,
)

from models.utility_functions import (
    skip_for_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report


def ttnn_bert_preprocess_inputs(
    input_ids,
    token_type_ids,
    attention_mask,
    **kwargs,
):
    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32)
    input_ids = ttnn.to_device(input_ids, kwargs["device"], memory_config=ttnn.L1_MEMORY_CONFIG)

    token_type_ids = ttnn.from_torch(token_type_ids, dtype=ttnn.uint32)
    token_type_ids = ttnn.to_device(token_type_ids, kwargs["device"], memory_config=ttnn.L1_MEMORY_CONFIG)

    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = F.pad(attention_mask, (0, 0, 0, 31, 0, 0, 0, kwargs["batch_size"] - 1))
        attention_mask = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16)
        attention_mask = ttnn.to_layout(attention_mask, ttnn.TILE_LAYOUT)
        attention_mask = ttnn.to_device(attention_mask, kwargs["device"], memory_config=ttnn.L1_MEMORY_CONFIG)

    return input_ids, token_type_ids, attention_mask


def convert_to_ttnn(torch_model, full_name):
    return True


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, transformers.models.bert.modeling_bert.BertSelfAttention):
        qkv_weight = torch.cat(
            [
                torch_model.query.weight,
                torch_model.key.weight,
                torch_model.value.weight,
            ],
            dim=0,
        )
        qkv_bias = torch.cat(
            [torch_model.query.bias, torch_model.key.bias, torch_model.value.bias],
            dim=0,
        )

        parameters = {"query_key_value": {}}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat16)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat16)
    return parameters


def get_expected_times(use_optimized_version):
    if use_optimized_version:
        expected_compile_time = 12
        expected_inference_time = 0.07
    else:
        expected_compile_time = 10
        expected_inference_time = 17
    return expected_compile_time, expected_inference_time


@skip_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("use_optimized_version", [True, False])
def test_performance(device, use_program_cache, model_name, batch_size, sequence_size, use_optimized_version):
    disable_persistent_kernel_cache()

    config = transformers.BertConfig.from_pretrained(model_name)

    # TODO(arakhmati): re-enable the line below once the issue with ttnn.embedding is fixed
    # torch_bert_input = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_bert_input = torch.randint(0, 1, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.zeros(1, sequence_size) if use_optimized_version else None

    # Run TT Model
    tt_model_name = "ttnn_" + ("optimized_" if use_optimized_version else "") + model_name
    parameters = preprocess_model_parameters(
        tt_model_name,
        initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        convert_to_ttnn=convert_to_ttnn,
        custom_preprocessor=custom_preprocessor if use_optimized_version else None,
        device=device,
    )

    bert_for_question_answering = (
        ttnn_optimized_bert_for_question_answering if use_optimized_version else ttnn_bert_for_question_answering
    )

    durations = []
    for _ in range(2):
        ttnn_bert_inputs = ttnn_bert_preprocess_inputs(
            torch_bert_input,
            torch_token_type_ids,
            torch_attention_mask,
            device=device,
            batch_size=batch_size,
        )

        start = time.time()
        tt_output = bert_for_question_answering(
            *ttnn_bert_inputs,
            parameters=parameters,
            num_heads=config.num_attention_heads,
        )
        tt_output = ttnn.from_device(tt_output)
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times(use_optimized_version)
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
