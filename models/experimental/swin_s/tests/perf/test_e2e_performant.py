# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
from loguru import logger
import ttnn
from models.experimental.swin_s.runner.performant_runner import SwinSPerformantRunner
from models.utility_functions import run_for_wormhole_b0
from models.experimental.swin_s.tt.common import get_mesh_mappers


def run_swin_s_inference(
    device,
    batch_size_per_device,
    resolution,
    act_dtype=ttnn.bfloat8_b,
    weight_dtype=ttnn.bfloat8_b,
):
    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)
    performant_runner = SwinSPerformantRunner(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        mesh_composer=outputs_mesh_composer,
    )
    performant_runner._capture_swins_trace_2cqs()

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    input_shape = (batch_size, 3, *resolution)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    inference_times = []
    for _ in range(10):
        t0 = time.time()
        _ = performant_runner.run(torch_input_tensor)
        t1 = time.time()
        inference_times.append(t1 - t0)

    performant_runner.release()

    inference_time_avg = round(sum(inference_times) / len(inference_times), 6)
    logger.info(
        f"ttnn_swin_batch_size: {batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size * num_devices/inference_time_avg)}"
    )


@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (512, 512),
    ],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 16998400, "num_command_queues": 2}], indirect=True
)
def test_e2e_performant(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    resolution,
):
    run_swin_s_inference(
        device,
        batch_size,
        resolution,
        act_dtype=ttnn.bfloat8_b,
        weight_dtype=ttnn.bfloat8_b,
    )


@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (512, 512),
    ],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 16998400, "num_command_queues": 2}], indirect=True
)
def test_e2e_performant_dp(
    mesh_device,
    batch_size,
    act_dtype,
    weight_dtype,
    resolution,
):
    run_swin_s_inference(
        mesh_device,
        batch_size,
        resolution,
        act_dtype=ttnn.bfloat8_b,
        weight_dtype=ttnn.bfloat8_b,
    )
