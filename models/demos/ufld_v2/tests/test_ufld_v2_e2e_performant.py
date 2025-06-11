# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger

from models.demos.ufld_v2.tests.ufld_v2_e2e_performant import UFLDPerformantRunner
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size",
    ((1),),
)
def test_run_ufldv2_trace_2cqs_inference(
    device,
    use_program_cache,
    batch_size,
    model_location_generator,
):
    performant_runner = UFLDPerformantRunner(
        device,
        batch_size,
    )
    performant_runner._capture_ufldv2_trace_2cqs()
    inference_time_iter = []
    for _ in range(10):
        t0 = time.time()
        _ = performant_runner._execute_ufldv2_trace_2cqs_inference()
        t1 = time.time()
        inference_time_iter.append(t1 - t0)
    performant_runner.release()
    inference_time_avg = round(sum(inference_time_iter) / len(inference_time_iter), 6)
    logger.info(
        f"ttnn_ufldv2_320x800_batch_size_{batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )
