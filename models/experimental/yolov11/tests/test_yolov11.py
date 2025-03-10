# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import ttnn
import time
import torch
import pytest
import torch.nn as nn
from loguru import logger
from models.utility_functions import is_wormhole_b0
from models.perf.perf_utils import prep_perf_report
from models.experimental.yolov11.tt import ttnn_yolov11
from models.experimental.yolov11.reference import yolov11
from models.experimental.yolov11.reference.yolov11 import attempt_load
from models.utility_functions import enable_persistent_kernel_cache, disable_persistent_kernel_cache
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.experimental.yolov11.tt.model_preprocessing import (
    create_yolov11_input_tensors,
    create_yolov11_model_parameters,
)

try:
    sys.modules["ultralytics"] = yolov11
    sys.modules["ultralytics.nn.tasks"] = yolov11
    sys.modules["ultralytics.nn.modules.conv"] = yolov11
    sys.modules["ultralytics.nn.modules.block"] = yolov11
    sys.modules["ultralytics.nn.modules.head"] = yolov11

except KeyError:
    print("models.experimental.yolov11.reference.yolov11 not found.")


def get_expected_times(name):
    base = {"yolov11": (130.70, 0.594)}
    return base[name]


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [(1)])
@pytest.mark.parametrize("input_tensor", [torch.rand((1, 3, 640, 640))], ids=["input_tensor"])
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True, False],
    ids=[
        "pretrained_weight_true",
        "pretrained_weight_false",
    ],
)
def test_yolov11(device, input_tensor, batch_size):
    disable_persistent_kernel_cache()
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=input_tensor.shape[0],
        input_channels=input_tensor.shape[1],
        input_height=input_tensor.shape[2],
        input_width=input_tensor.shape[3],
    )
    if use_pretrained_weight:
        torch_model = attempt_load("yolov11n.pt", map_location="cpu")
        state_dict = torch_model.state_dict()
    else:
        torch_model = yolov11.YoloV11()
        state_dict = torch_model.state_dict()

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)
    model = ttnn_yolov11.YoloV11(device, parameters)
    durations = []

    for i in range(2):
        start = time.time()
        ttnn_model_output = model(ttnn_input)
        end = time.time()
        durations.append(end - start)
        ttnn.deallocate(ttnn_model_output)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("yolov11")

    prep_perf_report(
        model_name="models/experimental/yolov11",
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
    "batch_size, expected_perf",
    [
        [1, 81.94],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov11(batch_size, expected_perf):
    subdir = "ttnn_yolov11"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 0

    command = f"pytest models/experimental/yolov11/demo/demo.py::test_demo"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_yolov11{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
