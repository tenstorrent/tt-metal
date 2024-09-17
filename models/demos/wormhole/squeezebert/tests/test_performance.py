# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import time
import ttnn
import torch
import pytest
import transformers
from loguru import logger
from models.perf.perf_utils import prep_perf_report
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.squeezebert.tt import ttnn_functional_squeezebert
from models.experimental.functional_common.attention_mask_functions import get_extended_attention_mask
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    is_wormhole_b0,
    skip_for_grayskull,
)


def synchronize_devices(device):
    devices = device.get_devices()
    for device in devices:
        ttnn.synchronize_device(device)


def get_expected_times(squeezebert):
    return {ttnn_functional_squeezebert: (29.29, 15.5)}[squeezebert]


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["squeezebert/squeezebert-uncased"])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("squeezebert", [ttnn_functional_squeezebert])
def test_performance(mesh_device, use_program_cache, model_name, sequence_size, squeezebert):
    disable_persistent_kernel_cache()

    num_iterations = 2
    batch_size = 8

    rf_model = transformers.SqueezeBertForQuestionAnswering.from_pretrained(model_name)
    config = transformers.SqueezeBertConfig.from_pretrained(model_name)
    state_dict = rf_model.state_dict()
    tt_model_name = f"ttnn_{model_name}_optimized"

    inputs_mesh_mapper = weights_mesh_mapper = output_mesh_composer = None
    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    if mesh_device_flag:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        batch_size = 16
        with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
            parameters = preprocess_model_parameters(
                model_name=tt_model_name,
                initialize_model=lambda: rf_model,
                custom_preprocessor=ttnn_functional_squeezebert.custom_preprocessor,
                device=mesh_device,
            )
    else:
        mesh_device = ttnn.open_device(device_id=0) if is_wormhole_b0() else mesh_device
        parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: rf_model,
            custom_preprocessor=ttnn_functional_squeezebert.custom_preprocessor,
            device=mesh_device,
        )

    input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.ones(batch_size, sequence_size)

    ttnn_squeezebert_inputs_on_cpu = ttnn_functional_squeezebert.preprocess_inputs(
        input_ids,
        torch_token_type_ids,
        position_ids,
        torch_attention_mask,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
    )

    start = time.time()
    ttnn_squeezebert_inputs = [
        (
            ttnn.to_device(tensor, device=mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG)
            if tensor is not None
            else tensor
        )
        for tensor in ttnn_squeezebert_inputs_on_cpu
    ]
    tt_output = squeezebert.squeezebert_for_question_answering(
        config,
        *ttnn_squeezebert_inputs,
        state_dict=state_dict,
        base_addr=f"transformer.",
        parameters=parameters,
        device=mesh_device,
        reader_patterns_cache={},
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
    )

    tt_output = ttnn.from_device(tt_output, blocking=False)
    synchronize_devices(mesh_device)
    end = time.time()
    inference_and_compile_time = end - start
    enable_persistent_kernel_cache()

    start = time.time()
    for _ in range(num_iterations):
        ttnn_squeezebert_inputs = [
            (
                ttnn.to_device(tensor, device=mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG)
                if tensor is not None
                else tensor
            )
            for tensor in ttnn_squeezebert_inputs_on_cpu
        ]
        tt_output = squeezebert.squeezebert_for_question_answering(
            config,
            *ttnn_squeezebert_inputs,
            state_dict=state_dict,
            base_addr=f"transformer.",
            parameters=parameters,
            device=mesh_device,
            reader_patterns_cache={},
            mesh_mapper=inputs_mesh_mapper,
            mesh_composer=output_mesh_composer,
        )
        tt_output = ttnn.from_device(tt_output, blocking=False)

    synchronize_devices(mesh_device)
    end = time.time()
    average_inference_time = (end - start) / num_iterations

    expected_compile_time, expected_inference_time = get_expected_times(squeezebert)
    prep_perf_report(
        model_name=tt_model_name,
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=average_inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - average_inference_time}")
    logger.info(f"Average Inference time: {average_inference_time}")
    logger.info(f"Samples per second: {1 / average_inference_time * batch_size}")

    assert (
        average_inference_time < expected_inference_time
    ), f"Expected inference time: {expected_inference_time} Actual inference time: {average_inference_time}"
