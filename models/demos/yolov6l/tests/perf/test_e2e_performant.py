# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0
from models.demos.yolov6l.common import YOLOV6L_L1_SMALL_SIZE
from models.demos.yolov6l.runner.performant_runner import YOLOv6lPerformantRunner
from models.demos.yolov6l.tt.common import get_mesh_mappers
from models.perf.perf_utils import prep_perf_report


def get_expected_times(name):
    base = {"yolov6l": (183.7, 0.0115)}
    return base[name]


def run_yolov6_inference(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    resolution,
    model_location_generator,
):
    disable_persistent_kernel_cache()

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

    performant_runner = YOLOv6lPerformantRunner(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
        mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        mesh_composer=output_mesh_composer,
    )

    input_shape = (batch_size, 3, 640, 640)
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
        f"Model: ttnn_yolov6 - batch_size: {batch_size}. One inference iteration time (sec): {inference_time}, FPS: {round(batch_size / inference_time)}"
    )

    expected_compile_time, expected_inference_time = get_expected_times("yolov6l")
    prep_perf_report(
        model_name="models/demos/yolov6l/",
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
    [{"l1_small_size": YOLOV6L_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
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
def test_perf_yolov6l(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    resolution,
    model_location_generator,
):
    run_yolov6_inference(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution,
        model_location_generator=model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV6L_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
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
def test_perf_yolov6l_dp(
    mesh_device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    resolution,
    model_location_generator,
):
    run_yolov6_inference(
        mesh_device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution,
        model_location_generator=model_location_generator,
    )
