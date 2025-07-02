# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.yolov9c.runner.performant_runner import YOLOv9PerformantRunner
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import run_for_wormhole_b0


def get_expected_times(name):
    base = {"yolov9c": (114.21, 40)}
    return base[name]


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 640),
    ],
)
@pytest.mark.parametrize(
    "model_task",
    [
        "segment",  # To run the test for instance segmentation
        "detect",  # To run the test for Object Detection
    ],
    ids=["segment", "detect"],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    model_task,
    model_location_generator,
    resolution,
):
    performant_runner = YOLOv9PerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_task=model_task,
        resolution=resolution,
        model_location_generator=None,
    )
    performant_runner._capture_yolov9_trace_2cqs()
    input_shape = (1, *resolution, 3)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    torch_input_tensor = F.pad(torch_input_tensor, (0, 29), mode="constant", value=0)
    tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    iterations = 32
    t0 = time.time()
    for _ in range(iterations):
        _ = performant_runner._execute_yolov9_trace_2cqs_inference(tt_inputs_host)
    ttnn.synchronize_device(device)
    t1 = time.time()

    performant_runner.release()

    inference_time = round((t1 - t0) / iterations, 4)
    inference_and_compile_time = inference_time  # Don't care about compile time

    expected_compile_time, expected_inference_time = get_expected_times("yolov9c")

    prep_perf_report(
        model_name="models/demos/yolov9c",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Inference time: {inference_time * 1000.0} ms")
    logger.info(f"Samples per second: {1 / inference_time * batch_size} fps")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize(
    "model_task",
    [
        "segment",  # To run the test for instance segmentation
        "detect",  # To run the test for Object Detection
    ],
    ids=["segment", "detect"],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device(model_task, batch_size):
    subdir = "ttnn_yolov9c"
    num_iterations = 1
    margin = 0.03
    enable_segment = model_task == "segment"
    expected_perf = 53.70 if enable_segment else 53.90

    command = f"pytest tests/ttnn/integration_tests/yolov9c/test_ttnn_yolov9c.py::test_yolov9c"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_functional_yolov9c{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
