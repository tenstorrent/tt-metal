# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import torch
import pytest
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import run_for_wormhole_b0
from models.utility_functions import enable_persistent_kernel_cache, disable_persistent_kernel_cache
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report

from models.experimental.functional_yolov9c.tt import ttnn_yolov9c
from models.experimental.functional_yolov9c.reference import yolov9c
from models.experimental.functional_yolov9c.tt.model_preprocessing import (
    create_yolov9c_input_tensors,
    create_yolov9c_model_parameters,
)


def get_expected_times(name):
    base = {"yolov9c": (114.21, 1.082)}
    return base[name]


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
def test_perf(device, use_weights_from_ultralytics):
    disable_persistent_kernel_cache()
    torch_input, ttnn_input = create_yolov9c_input_tensors(device, model=True)
    batch_size = torch_input.shape[0]
    torch_model = yolov9c.YoloV9()

    if use_weights_from_ultralytics:
        pretrained_model = YOLO("yolov9c.pt")
        torch_model.load_state_dict(pretrained_model.state_dict(), strict=False)

    new_state_dict = {
        name: param for name, param in torch_model.state_dict().items() if isinstance(param, torch.FloatTensor)
    }

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    parameters = create_yolov9c_model_parameters(torch_model, torch_input, device=device)
    ttnn_model = ttnn_yolov9c.YoloV9(device, parameters)

    durations = []

    for i in range(2):
        start = time.time()
        ttnn_model_output = ttnn_model(ttnn_input)
        end = time.time()
        durations.append(end - start)
        ttnn.deallocate(ttnn_model_output)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("yolov9c")

    prep_perf_report(
        model_name="models/experimental/functional_yolov9c",
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
        [1, 76.27],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov9c(batch_size, expected_perf):
    subdir = "ttnn_yolov9c"
    num_iterations = 1
    margin = 0.03

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
