# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import time
import torch
import pytest
import transformers

from loguru import logger
from models.demos.wormhole.roberta.tt import ttnn_optimized_roberta
from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import (
    is_grayskull,
    is_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    run_for_wormhole_b0,
)
from models.perf.perf_utils import prep_perf_report


def get_expected_times(bert):
    return {
        ttnn_optimized_roberta: (30.010, 0.09),
    }[bert]


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["deepset/roberta-large-squad2"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("bert", [ttnn_optimized_roberta])
def test_performance(mesh_device, use_program_cache, model_name, batch_size, sequence_size, bert):
    disable_persistent_kernel_cache()

    config = transformers.RobertaConfig.from_pretrained(model_name)
    config.use_dram = True

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = batch_size * 2 if mesh_device_flag else batch_size

    input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.zeros(1, sequence_size) if bert == ttnn_optimized_roberta else None
    torch_position_ids = create_position_ids_from_input_ids(input_ids=input_ids, padding_idx=config.pad_token_id)
    tt_model_name = f"ttnn_{model_name}_optimized"

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: transformers.RobertaForQuestionAnswering.from_pretrained(
                model_name, torchscript=False
            ),
            custom_preprocessor=ttnn_optimized_roberta.custom_preprocessor,
            device=mesh_device,
        )

    durations = []
    for _ in range(2):
        ttnn_bert_inputs = bert.preprocess_inputs(
            input_ids,
            torch_token_type_ids,
            torch_position_ids,
            torch_attention_mask,
            device=mesh_device,
            inputs_mesh_mapper=inputs_mesh_mapper,
        )

        start = time.time()
        tt_output = bert.bert_for_question_answering(
            config,
            *ttnn_bert_inputs,
            parameters=parameters,
            name="roberta",
        )
        tt_output = ttnn.from_device(tt_output)
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times(bert)
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

    assert (
        inference_time < expected_inference_time
    ), f"Expected inference time: {expected_inference_time} Actual inference time: {inference_time}"
