# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import pytest
import ttnn
import time

from loguru import logger
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from transformers import BertForQuestionAnswering
from models.demos.wormhole.bert_tiny.tt.bert_tiny import bert_for_question_answering
from models.utility_functions import is_wormhole_b0, skip_for_grayskull


def get_expected_times(bert_tiny):
    return (38.5, 1.6)


@skip_for_grayskull()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("model_name", ["mrm8488/bert-tiny-finetuned-squadv2"])
def test_perf_bert_tiny(
    mesh_device,
    sequence_size,
    model_name,
    model_location_generator,
    reset_seeds,
):
    disable_persistent_kernel_cache()
    model_name = str(model_location_generator(model_name, model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    config = hugging_face_reference_model.config
    pytorch_model = hugging_face_reference_model

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = 16 if mesh_device_flag else 8

    torch_bert_input = torch.randint(0, 100, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.zeros(1, sequence_size)

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: pytorch_model,
            device=mesh_device,
            convert_to_ttnn=lambda *_: True,
        )

    ttnn_bert_inputs = ttnn.from_torch(
        torch_bert_input, dtype=ttnn.uint32, mesh_mapper=inputs_mesh_mapper, device=mesh_device
    )
    ttnn_token_type_ids = ttnn.from_torch(
        torch_token_type_ids, dtype=ttnn.uint32, mesh_mapper=inputs_mesh_mapper, device=mesh_device
    )
    ttnn_position_ids = ttnn.from_torch(
        torch_position_ids, dtype=ttnn.uint32, mesh_mapper=inputs_mesh_mapper, device=mesh_device
    )
    ttnn_attention_mask = ttnn.from_torch(
        torch_attention_mask,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )
    durations = []
    for i in range(2):
        start = time.time()
        ttnn_output = bert_for_question_answering(
            config,
            input_ids=ttnn_bert_inputs,
            token_type_ids=ttnn_token_type_ids,
            position_ids=ttnn_position_ids,
            attention_mask=ttnn_attention_mask,
            parameters=parameters,
            device=mesh_device,
        )
        output = ttnn.from_device(ttnn_output)

        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("bert_tiny")
    prep_perf_report(
        model_name="bert_tiny",
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
    logger.info("Exit Bert-Tiny perf test")


@skip_for_grayskull()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        (16, 6292.78),
    ],
)
def test_perf_device_bare_metal(batch_size, expected_perf):
    subdir = "ttnn_bert_tiny"
    num_iterations = 1
    margin = 0.03

    command = f"pytest tests/ttnn/integration_tests/bert_tiny/test_bert_tiny_wh.py::test_bert_for_question_answering"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    prep_device_perf_report(
        model_name=f"ttnn_bert_tiny{batch_size}",
        batch_size=batch_size if mesh_device_flag else 8,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
