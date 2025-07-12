# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov4.runner.performant_runner import YOLOv4PerformantRunner
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 40960, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution, expected_inference_throughput",
    [
        ((320, 320), 103),
        ((640, 640), 52),
    ],
)
def test_perf_e2e_yolov4(device, batch_size, act_dtype, weight_dtype, resolution, expected_inference_throughput):
    performant_runner = YOLOv4PerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=None,
    )

    input_shape = (1, 3, *resolution)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    iterations = 32
    t0 = time.time()
    for _ in range(iterations):
        _ = performant_runner.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()

    performant_runner.release()

    inference_time_avg = round((t1 - t0) / iterations, 4)
    throughput_avg = round(batch_size / inference_time_avg, 2)
    logger.info(f"average inference time: {inference_time_avg * 1000} ms, average throughput: {throughput_avg} fps")

    expected_inference_time = batch_size / expected_inference_throughput
    prep_perf_report(
        model_name="yolov4",
        batch_size=batch_size,
        inference_and_compile_time=inference_time_avg,
        inference_time=inference_time_avg,
        expected_compile_time=1,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    assert (
        throughput_avg >= expected_inference_throughput
    ), f"Expected end-to-end performance to exceed {expected_inference_throughput} fps but was {throughput_avg} fps"


@pytest.mark.parametrize(
    "batch_size, model_name, expected_perf",
    [
        (1, "yolov4", 93.6),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov4(batch_size, model_name, expected_perf):
    subdir = model_name
    num_iterations = 1
    margin = 0.03

    command = f"pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[1-pretrained_weight_false-0]"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_functional_{model_name}_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
