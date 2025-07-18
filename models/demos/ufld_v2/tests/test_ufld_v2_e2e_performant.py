# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

from models.demos.ufld_v2.runner.performant_runner import UFLDPerformantRunner
from models.utility_functions import run_for_wormhole_b0
from tests.ttnn.integration_tests.ufld_v2.test_ttnn_ufld_v2 import get_mesh_mappers


def run_ufldv2_e2e(device, batch_size_per_device, model_location_generator, height=320, width=800):
    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    input_shape = (batch_size, 3, height, width)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    performant_runner = UFLDPerformantRunner(
        device,
        batch_size,
        torch_input_tensor=torch_input_tensor,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
    )
    performant_runner._capture_ufldv2_trace_2cqs()
    num_iter = 1000
    inference_time_iter = []
    t0 = time.time()
    for _ in range(num_iter):
        output_tensor = performant_runner.run(torch_input_tensor)  # workaround - _execute_ufldv2_trace_2cqs_inference()
    t1 = time.time()
    performant_runner.release()
    inference_time_avg = round((t1 - t0) / num_iter, 6)
    logger.info(
        f"ttnn_ufldv2_320x800_batch_size_{batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round((batch_size_per_device * num_devices)/inference_time_avg)}"
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size",
    ((1),),
)
def test_ufldv2_e2e_performant(device, batch_size, model_location_generator):
    run_ufldv2_e2e(device, batch_size, model_location_generator)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((1),),
)
def test_ufldv2_e2e_performant_dp(mesh_device, batch_size_per_device, model_location_generator):
    run_ufldv2_e2e(mesh_device, batch_size_per_device, model_location_generator)
