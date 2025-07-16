# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.demos.yolov10x.reference.yolov10x import YOLOv10
from models.demos.yolov10x.tt.model_preprocessing import create_yolov10x_input_tensors, create_yolov10x_model_parameters
from models.demos.yolov10x.tt.yolov10x import TtnnYolov10
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    run_for_wormhole_b0,
)


def get_expected_times(name):
    base = {"yolov10x": (106.10, 0.92)}
    return base[name]


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        "True",
    ],
)
def test_perf(device, use_weights_from_ultralytics):
    disable_persistent_kernel_cache()
    torch_input, ttnn_input = create_yolov10x_input_tensors(device)
    batch_size = torch_input.shape[0]
    state_dict = None

    if use_weights_from_ultralytics:
        torch_model = YOLO("yolov10x.pt")
        state_dict = torch_model.state_dict()

    torch_model = YOLOv10()
    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    parameters = create_yolov10x_model_parameters(torch_model, torch_input, device)

    torch_model_output = torch_model(torch_input)[0]
    ttnn_model = TtnnYolov10(
        device=device,
        parameters=parameters,
        conv_pt=parameters,
    )

    durations = []

    for i in range(2):
        torch_input, ttnn_input = create_yolov10x_input_tensors(device)
        start = time.time()
        ttnn_model_output = ttnn_model(ttnn_input)
        end = time.time()
        durations.append(end - start)
        ttnn.deallocate(ttnn_model_output)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("yolov10x")

    prep_perf_report(
        model_name="models/demos/yolov10x",
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


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 44.5],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov10x(batch_size, expected_perf):
    subdir = "ttnn_yolov10"
    num_iterations = 1
    margin = 0.03

    command = f"pytest tests/ttnn/integration_tests/yolov10x/test_ttnn_yolov10x.py::test_yolov10x"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_yolov10x{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
