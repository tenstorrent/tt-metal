# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
RF-DETR Medium end-to-end performant tests.

Two test modes:
  1. No-trace baseline:  plain forward, 1 command queue
  2. 2CQ overlapped:     H2D on CQ1 overlapped with compute on CQ0

RF-DETR has a host barrier (two-stage top-K) between backbone+projector
and decoder+heads, so full trace capture is not applicable. The 2CQ test
measures the benefit of overlapping input transfer with computation.

Usage:
    pytest models/experimental/rfdetr_medium/tests/perf/test_e2e_performant.py -v -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn

from models.perf.perf_utils import prep_perf_report
from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.rfdetr_medium.common import (
    RESOLUTION,
    RFDETR_MEDIUM_L1_SMALL_SIZE,
)


def get_expected_times(name):
    return {
        "rfdetr_medium_no_trace": (60.0, 0.075),
        "rfdetr_medium_2cq": (60.0, 0.070),
    }[name]


def run_e2e_no_trace(device, batch_size, num_iterations):
    """Baseline: no trace, single command queue, plain forward."""
    from models.experimental.rfdetr_medium.runner.performant_runner import RFDETRPerformantRunner

    runner = RFDETRPerformantRunner(device, batch_size)
    torch_input = torch.randn(batch_size, 3, RESOLUTION, RESOLUTION)

    logger.info("Warmup (2 iterations, no trace)...")
    for _ in range(2):
        runner.run_no_trace(torch_input)
        ttnn.synchronize_device(device)

    logger.info(f"Running {num_iterations} iterations (no trace)...")
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(num_iterations):
        runner.run_no_trace(torch_input)
    ttnn.synchronize_device(device)
    t1 = time.time()

    inference_time_avg = (t1 - t0) / num_iterations
    fps = batch_size / inference_time_avg

    logger.info(f"RF-DETR Medium no-trace: batch={batch_size}, " f"avg={inference_time_avg*1000:.2f} ms, FPS={fps:.1f}")
    return inference_time_avg, fps


def run_e2e_2cq(device, batch_size, num_iterations):
    """2 command queue overlapped H2D + compute."""
    from models.experimental.rfdetr_medium.runner.performant_runner import RFDETRPerformantRunner

    runner = RFDETRPerformantRunner(device, batch_size)
    torch_input = torch.randn(batch_size, 3, RESOLUTION, RESOLUTION)

    logger.info("Warmup (2CQ)...")
    runner._warmup_2cqs(num_warmup=2)

    logger.info(f"Running {num_iterations} iterations (2CQ)...")
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(num_iterations):
        runner.run(torch_input)
    ttnn.synchronize_device(device)
    t1 = time.time()

    runner.release()

    inference_time_avg = (t1 - t0) / num_iterations
    fps = batch_size / inference_time_avg

    logger.info(f"RF-DETR Medium 2CQ: batch={batch_size}, " f"avg={inference_time_avg*1000:.2f} ms, FPS={fps:.1f}")
    return inference_time_avg, fps


# --- No-trace baseline test ---


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": RFDETR_MEDIUM_L1_SMALL_SIZE}],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [10])
@pytest.mark.parametrize("batch_size", [1])
@run_for_wormhole_b0()
def test_rfdetr_medium_e2e_no_trace(device, batch_size, num_iterations):
    """Baseline performance: no trace, 1CQ."""
    inference_time, fps = run_e2e_no_trace(device, batch_size, num_iterations)

    expected_compile_time, expected_inference_time = get_expected_times("rfdetr_medium_no_trace")
    prep_perf_report(
        model_name="rfdetr_medium_no_trace",
        batch_size=batch_size,
        inference_and_compile_time=inference_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=f"batch_{batch_size}_no_trace",
    )


# --- 2CQ overlapped test ---


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": RFDETR_MEDIUM_L1_SMALL_SIZE, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [10])
@pytest.mark.parametrize("batch_size", [1])
@run_for_wormhole_b0()
def test_rfdetr_medium_e2e_2cq(device, batch_size, num_iterations):
    """Performance with 2 command queues: overlapped H2D + compute."""
    inference_time, fps = run_e2e_2cq(device, batch_size, num_iterations)

    expected_compile_time, expected_inference_time = get_expected_times("rfdetr_medium_2cq")
    prep_perf_report(
        model_name="rfdetr_medium_2cq",
        batch_size=batch_size,
        inference_and_compile_time=inference_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=f"batch_{batch_size}_2cq",
    )
