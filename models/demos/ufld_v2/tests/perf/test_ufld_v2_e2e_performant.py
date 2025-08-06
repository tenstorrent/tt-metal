# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger

import ttnn
from models.demos.ufld_v2.runner.performant_runner import UFLDPerformantRunner
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size",
    ((1),),
)
def test_run_ufldv2_trace_2cqs_inference(
    device,
    batch_size,
    model_location_generator,
):
    performant_runner = UFLDPerformantRunner(
        device,
        batch_size,
    )
    performant_runner._capture_ufldv2_trace_2cqs()
    num_iter = 1000
    inference_time_iter = []
    t0 = time.time()
    for _ in range(num_iter):
        output_tensor = performant_runner.run()
    ttnn.synchronize_device(device)
    t1 = time.time()
    performant_runner.release()
    inference_time_avg = round((t1 - t0) / num_iter, 6)
    logger.info(
        f"ttnn_ufldv2_320x800_batch_size_{batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )
