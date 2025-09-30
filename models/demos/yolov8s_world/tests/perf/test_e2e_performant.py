# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov8s_world.common import YOLOV8SWORLD_L1_SMALL_SIZE
from models.demos.yolov8s_world.runner.performant_runner import YOLOv8sWorldPerformantRunner
from models.perf.perf_utils import prep_perf_report


def get_expected_times(name):
    base = {"yolov8s_world": (183.7, 0.038)}
    return base[name]


def run_yolov8s_world_inference(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    performant_runner = YOLOv8sWorldPerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
        inputs_mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        outputs_mesh_composer=outputs_mesh_composer,
    )
    performant_runner._capture_yolov8s_world_trace_2cqs()

    input_shape = (batch_size, 3, *resolution)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    t0 = time.time()
    for _ in range(10):
        _ = performant_runner.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()

    performant_runner.release()
    inference_time = round((t1 - t0) / 10, 6)

    inference_and_compile_time = inference_time

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(
        f"Model: ttnn_yolov8s - batch_size: {batch_size}. One inference iteration time (sec): {inference_time}, FPS: {round(batch_size / inference_time)}"
    )

    expected_compile_time, expected_inference_time = get_expected_times("yolov8s_world")
    prep_perf_report(
        model_name="models/demos/yolov8s_world",
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
    "device_params",
    [{"l1_small_size": YOLOV8SWORLD_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 640),
    ],
)
def test_perf_yolov8s_world(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    run_yolov8s_world_inference(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8SWORLD_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 640),
    ],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf_yolov8s_world_dp(
    mesh_device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    run_yolov8s_world_inference(
        mesh_device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )
