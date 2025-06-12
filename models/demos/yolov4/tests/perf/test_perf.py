# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov4.common import load_torch_model
from models.demos.yolov4.tt.model_preprocessing import create_yolov4_model_parameters
from models.demos.yolov4.tt.yolov4 import TtYOLOv4
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import disable_persistent_kernel_cache, profiler


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, expected_compile_time, expected_inference_time",
    [
        ((1, 320, 320, 3), 70, 0.5),
        ((1, 640, 640, 3), 70, 0.6),
    ],
)
def test_yolov4(
    device,
    input_shape,
    expected_compile_time,
    expected_inference_time,
    model_location_generator,
):
    disable_persistent_kernel_cache()
    profiler.clear()

    batch_size = input_shape[0]
    resolution = input_shape[1:3]
    torch_model = load_torch_model(model_location_generator)
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_input = torch_input_tensor.permute(0, 3, 1, 2).float()
    parameters = create_yolov4_model_parameters(torch_model, torch_input, resolution, device)

    ttnn_input = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    ttnn_model = TtYOLOv4(parameters, device)

    logger.info(f"Compiling model with warmup run")
    profiler.start(f"inference_and_compile_time")
    ttnn_output_tensor = ttnn_model(ttnn_input)
    ttnn.deallocate(ttnn_output_tensor[0])
    ttnn.deallocate(ttnn_output_tensor[1])

    profiler.end(f"inference_and_compile_time")

    inference_and_compile_time = profiler.get("inference_and_compile_time")
    logger.info(
        f"Model with input resolution {resolution} compiled with warmup run in {(inference_and_compile_time):.2f} s"
    )

    iterations = 16

    outputs = []
    logger.info(f"Running inference for {iterations} iterations")
    for idx in range(iterations):
        profiler.start("inference_time")
        profiler.start(f"inference_time_{idx}")
        ttnn_output_tensor = ttnn_model(ttnn_input)
        ttnn.deallocate(ttnn_output_tensor[0])
        ttnn.deallocate(ttnn_output_tensor[1])
        profiler.end(f"inference_time_{idx}")
        profiler.end("inference_time")

    mean_inference_time = profiler.get("inference_time")
    inference_time = profiler.get(f"inference_time_{iterations - 1}")
    compile_time = inference_and_compile_time - inference_time
    logger.info(f"Model compilation of resolution {resolution} took {compile_time:.1f} s")
    logger.info(
        f"Inference time on last iterations for resolution: {resolution} was completed in {(inference_time * 1000.0):.2f} ms"
    )
    logger.info(
        f"Mean inference time for {batch_size} (batch), resolution {resolution} images was {(mean_inference_time * 1000.0):.2f} ms ({batch_size / mean_inference_time:.2f} fps)"
    )

    prep_perf_report(
        model_name="yolov4",
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


@pytest.mark.parametrize(
    "batch_size, model_name, expected_perf",
    [
        (1, "yolov4", 87.8),
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
