# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import torch
from loguru import logger
import pytest

import ttnn
from models.perf.perf_utils import prep_perf_report
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import is_wormhole_b0, run_for_wormhole_b0
from models.utility_functions import enable_persistent_kernel_cache, disable_persistent_kernel_cache
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.experimental.yolov8s_world.tt.ttnn_yolov8s_world_utils import (
    create_custom_preprocessor,
    attempt_load,
    move_to_device,
)
from models.experimental.yolov8s_world.tt.ttnn_yolov8s_world import (
    TtYOLOWorld,
)
from models.experimental.yolov8s_world.reference import yolov8s_world


def get_expected_times(name):
    base = {"yolov8s_world": (183.7, 0.4)}
    return base[name]


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
        True,
    ],
)
def test_perf(device, use_pretrained_weight, use_program_cache):
    disable_persistent_kernel_cache()
    torch_input = torch.randn(1, 3, 640, 640)

    ttnn_input = torch_input.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(
        ttnn_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    batch_size = torch_input.shape[0]

    state_dict = None
    if use_pretrained_weight:
        weights_torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
        torch_model = yolov8s_world.YOLOWorld(model_torch=weights_torch_model)

        state_dict = weights_torch_model.state_dict()
        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
            new_state_dict[name1] = parameter2

        torch_model.load_state_dict(new_state_dict)
        torch_model = torch_model.model
    else:
        torch_model = yolov8s_world.YOLOWorld()
        state_dict = torch_model.state_dict()
        torch_model = torch_model.model

    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    for i in [12, 15, 19, 22]:
        parameters["model"][i]["attn"]["gl"]["weight"] = ttnn.to_device(
            parameters["model"][i]["attn"]["gl"]["weight"], device=device
        )
        parameters["model"][i]["attn"]["gl"]["bias"] = ttnn.to_device(
            parameters["model"][i]["attn"]["gl"]["bias"], device=device
        )
        parameters["model"][i]["attn"]["bias"] = ttnn.to_device(parameters["model"][i]["attn"]["bias"], device=device)

    parameters["model"][16] = move_to_device(parameters["model"][16], device)

    parameters["model"][23]["cv4"] = move_to_device(parameters["model"][23]["cv4"], device)

    ttnn_model = TtYOLOWorld(
        device,
        parameters,
    )

    durations = []

    for i in range(2):
        start = time.time()
        ttnn_model_output, ttnn_model_output_x = ttnn_model(ttnn_input)
        end = time.time()
        durations.append(end - start)
        ttnn.deallocate(ttnn_model_output)
        for i in range(len(ttnn_model_output_x)):
            ttnn.deallocate(ttnn_model_output_x[i])
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("yolov8s_world")

    prep_perf_report(
        model_name="models/experimental/yolov8s_world",
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
        [1, 79.2],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov9c(batch_size, expected_perf):
    subdir = "ttnn_yolov8s_world"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 0

    command = f"pytest tests/ttnn/integration_tests/yolov8s_world/test_ttnn_yolov8s_world.py::test_YoloModel"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_yolov8s_world{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
