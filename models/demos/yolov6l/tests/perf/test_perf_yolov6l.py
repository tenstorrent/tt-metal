# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov6l.runner.performant_runner import YOLOv6lPerformantRunner
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import disable_persistent_kernel_cache, is_wormhole_b0, run_for_wormhole_b0


def get_expected_times(name):
    base = {"yolov6l": (183.7, 0.014)}
    return base[name]


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 640),
    ],
)
@pytest.mark.models_performance_bare_metal
def test_perf_yolov6l(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    resolution,
):
    disable_persistent_kernel_cache()
    performant_runner = YOLOv6lPerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=None,
    )
    performant_runner._capture_yolov6l_trace_2cqs()
    input_shape = (1, 3, *resolution)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    iterations = 32
    t0 = time.time()
    for _ in range(iterations):
        _ = performant_runner.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()

    performant_runner.release()

    inference_time = round((t1 - t0) / iterations, 6)
    inference_and_compile_time = inference_time  # Don't care about compile time

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")

    expected_compile_time, expected_inference_time = get_expected_times("yolov6l")
    prep_perf_report(
        model_name="models/demos/yolov6l",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 77],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov6l(batch_size, expected_perf):
    subdir = "ttnn_yolov6l"
    num_iterations = 1
    margin = 0.05
    expected_perf = expected_perf if is_wormhole_b0() else 0

    command = f"pytest tests/ttnn/integration_tests/yolov6l/test_ttnn_yolov6l.py::test_yolov6l"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_yolov6l{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
