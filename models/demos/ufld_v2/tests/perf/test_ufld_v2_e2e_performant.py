# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.ufld_v2.common import UFLD_V2_L1_SMALL_SIZE
from models.demos.ufld_v2.runner.performant_runner import UFLDPerformantRunner


def run_ufldv2_e2e(device, batch_size_per_device, model_location_generator, height=320, width=800, channels=3):
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    input_shape = (batch_size, channels, height, width)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    performant_runner = UFLDPerformantRunner(
        device,
        device_batch_size=batch_size_per_device,
        torch_input_tensor=torch_input_tensor,
        model_location_generator=model_location_generator,
    )
    performant_runner._capture_ufldv2_trace_2cqs()
    num_iter = 1000
    inference_time_iter = []
    t0 = time.time()
    for _ in range(num_iter):
        output_tensor = performant_runner.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()
    performant_runner.release()
    inference_time_avg = round((t1 - t0) / num_iter, 6)
    logger.info(
        f"ttnn_ufldv2_320x800_batch_size_{batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round((batch_size_per_device * num_devices)/inference_time_avg)}"
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": UFLD_V2_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    ((1),),
)
def test_ufldv2_e2e_performant(device, batch_size, model_location_generator):
    run_ufldv2_e2e(device, batch_size, model_location_generator)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": UFLD_V2_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((1),),
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_ufldv2_e2e_performant_dp(mesh_device, batch_size_per_device, model_location_generator):
    run_ufldv2_e2e(mesh_device, batch_size_per_device, model_location_generator)
