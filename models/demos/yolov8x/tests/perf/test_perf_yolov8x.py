# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.demos.yolov8x.reference import yolov8x
from models.demos.yolov8x.tt.ttnn_yolov8x import TtYolov8xModel
from models.demos.yolov8x.tt.ttnn_yolov8x_utils import custom_preprocessor
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import disable_persistent_kernel_cache, profiler


def get_expected_times(name):
    base = {"yolov8x": (128.267, 0.56)}
    return base[name]

@pytest.mark.skip(reason="https://github.com/tenstorrent/tt-metal/issues/24706")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [torch.rand((1, 3, 640, 640))], ids=["input_tensor"])
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
def test_yolov8x(device, input_tensor, use_weights_from_ultralytics):
    disable_persistent_kernel_cache()
    profiler.clear()
    batch_size = input_tensor.shape[0]
    if use_weights_from_ultralytics:
        torch_model = YOLO("yolov8x.pt")
        torch_model = torch_model.model
        torch_model.eval()
        state_dict = torch_model.state_dict()
    else:
        torch_model = yolov8x.DetectionModel()
        state_dict = torch_model.state_dict()
    parameters = custom_preprocessor(device, state_dict)

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_model = TtYolov8xModel(device, parameters)
    logger.info(f"Compiling model with warmup run")
    profiler.start(f"inference_and_compile_time")
    ttnn_output_tensor = ttnn_model(ttnn_input)

    profiler.end(f"inference_and_compile_time")

    inference_and_compile_time = profiler.get("inference_and_compile_time")
    logger.info(f"Model compiled with warmup run in {(inference_and_compile_time):.2f} s")

    iterations = 16
    outputs = []
    logger.info(f"Running inference for {iterations} iterations")
    for idx in range(iterations):
        profiler.start("inference_time")
        profiler.start(f"inference_time_{idx}")
        ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        ttnn_output_tensor = ttnn_model(ttnn_input)

        profiler.end(f"inference_time_{idx}")
        profiler.end("inference_time")

    mean_inference_time = profiler.get("inference_time")
    inference_time = profiler.get(f"inference_time_{iterations - 1}")
    compile_time = inference_and_compile_time - inference_time
    logger.info(f"Model compilation took {compile_time:.1f} s")
    logger.info(f"Inference time on last iterations was completed in {(inference_time * 1000.0):.2f} ms")
    logger.info(
        f"Mean inference time for {batch_size} (batch) images was {(mean_inference_time * 1000.0):.2f} ms ({batch_size / mean_inference_time:.2f} fps)"
    )

    expected_compile_time, expected_inference_time = get_expected_times("yolov8x")

    prep_perf_report(
        model_name="models/demos/functional_yolov8x",
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
    logger.info("Exit Yolov8x perf test")


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 45],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov8x(batch_size, expected_perf):
    subdir = "ttnn_yolov8x"
    num_iterations = 1
    margin = 0.03
    command = f"pytest tests/ttnn/integration_tests/yolov8x/test_yolov8x.py::test_yolov8x_640[use_weights_from_ultralytics=True-input_tensor1-0]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_functional_yolov8x{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
