# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger

import ttnn
from models.demos.yolov9c.demo.demo_utils import load_torch_model
from models.demos.yolov9c.tt import ttnn_yolov9c
from models.demos.yolov9c.tt.model_preprocessing import create_yolov9c_input_tensors, create_yolov9c_model_parameters
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    run_for_wormhole_b0,
)


def get_expected_times(name):
    base = {"yolov9c": (114.21, 1.3)}
    return base[name]


def dealloc_output(output_tensor):
    ttnn.deallocate(output_tensor[0])
    for t in output_tensor[1]:
        if isinstance(t, list):
            for sub_t in t:
                ttnn.deallocate(sub_t)
        else:
            ttnn.deallocate(t)


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        # "False",
        "True",
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
def test_perf(device, model_task, use_weights_from_ultralytics):
    disable_persistent_kernel_cache()
    enable_segment = model_task == "segment"
    # https://github.com/tenstorrent/tt-metal/issues/23288
    device.disable_and_clear_program_cache()

    torch_input, ttnn_input = create_yolov9c_input_tensors(device, model=True)

    batch_size = torch_input.shape[0]
    torch_model = load_torch_model(use_weights_from_ultralytics=use_weights_from_ultralytics, model_task=model_task)
    parameters = create_yolov9c_model_parameters(torch_model, torch_input, device=device)
    ttnn_model = ttnn_yolov9c.YoloV9(device, parameters, enable_segment=enable_segment)

    durations = []

    for i in range(2):
        start = time.time()
        torch_input, ttnn_input = create_yolov9c_input_tensors(device, model=True)
        ttnn_model_output = ttnn_model(ttnn_input)
        end = time.time()
        durations.append(end - start)
        dealloc_output(ttnn_model_output)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

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

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")


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
def test_perf_device_bare_metal_yolov9c(model_task, batch_size):
    subdir = "ttnn_yolov9c"
    num_iterations = 1
    margin = 0.03
    enable_segment = model_task == "segment"
    expected_perf = 30.6 if enable_segment else 30.7

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
