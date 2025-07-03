# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0
import sys
import time

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import models.demos.yolov7.reference.yolov7_model as yolov7_model
import models.demos.yolov7.reference.yolov7_utils as yolov7_utils
import ttnn
from models.demos.yolov7.reference.model import Yolov7_model
from models.demos.yolov7.reference.yolov7_utils import download_yolov7_weights
from models.demos.yolov7.tt.ttnn_yolov7 import ttnn_yolov7
from models.demos.yolov7.ttnn_yolov7_utils import create_custom_preprocessor, create_yolov7_input_tensors, load_weights
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    run_for_wormhole_b0,
)

sys.modules["models.common"] = yolov7_utils
sys.modules["models.yolo"] = yolov7_model


def get_expected_times(name):
    base = {"yolov7": (124, 1.082)}
    return base[name]


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_perf(device, reset_seeds):
    disable_persistent_kernel_cache()

    torch_model = Yolov7_model()

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    ds_state_dict = {k: v for k, v in torch_model.state_dict().items()}
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input, ttnn_input = create_yolov7_input_tensors(device, model=True)
    batch_size = torch_input.shape[0]
    weights_path = "tests/ttnn/integration_tests/yolov7/yolov7.pt"
    weights_path = download_yolov7_weights(weights_path)
    load_weights(torch_model, weights_path)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )

    nx_ny = [80, 40, 20]
    grid_tensors = []
    for i in range(3):
        yv, xv = torch.meshgrid([torch.arange(nx_ny[i]), torch.arange(nx_ny[i])])
        grid_tensors.append(torch.stack((xv, yv), 2).view((1, 1, nx_ny[i], nx_ny[i], 2)).float())

    ttnn_model = ttnn_yolov7(device, parameters, grid_tensors)

    durations = []

    for i in range(2):
        start = time.time()
        torch_input, ttnn_input = create_yolov7_input_tensors(device, model=True)
        ttnn_model_output = ttnn_model(ttnn_input)[0]
        end = time.time()
        durations.append(end - start)
        ttnn.deallocate(ttnn_model_output)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("yolov7")

    prep_perf_report(
        model_name="models/demos/yolov7",
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
        [1, 69.9],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov7(batch_size, expected_perf):
    subdir = "ttnn_yolov7"
    num_iterations = 1
    margin = 0.03

    command = f"pytest tests/ttnn/integration_tests/yolov7/test_ttnn_yolov7.py::test_yolov7"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_functional_yolov7{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
