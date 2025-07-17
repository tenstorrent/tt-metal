# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov4.common import get_mesh_mappers
from models.demos.yolov4.runner.performant_runner import YOLOv4PerformantRunner
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import run_for_wormhole_b0


def run_perf_e2e_yolov4(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    expected_inference_throughput,
):
    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)

    performant_runner = YOLOv4PerformantRunner(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=None,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
    )

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    input_shape = (batch_size_per_device * num_devices, 3, *resolution)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper)

    inference_times = []
    for _ in range(10):
        t0 = time.time()
        _ = performant_runner.run(torch_input_tensor)
        t1 = time.time()
        inference_times.append(t1 - t0)

    performant_runner.release()
    inference_time_avg = sum(inference_times) / len(inference_times)
    logger.info(
        f"Model: ttnn_yolov4 - batch_size: {batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {(batch_size * num_devices)/inference_time_avg}"
    )

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


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 40960, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution, expected_inference_throughput",
    [((320, 320), 103), ((640, 640), 46)],
)
def test_e2e_performant(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    expected_inference_throughput,
):
    run_perf_e2e_yolov4(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
        expected_inference_throughput,
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 40960, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution, expected_inference_throughput",
    [((320, 320), 103), ((640, 640), 46)],
)
def test_e2e_performant_dp(
    mesh_device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    expected_inference_throughput,
):
    run_perf_e2e_yolov4(
        mesh_device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
        expected_inference_throughput,
    )
