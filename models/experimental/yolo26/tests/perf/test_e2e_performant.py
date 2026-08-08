# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLO26 End-to-End Performant Tests.

High-level performance tests for YOLO26 using trace 2CQ execution.
"""

import time
import pytest
import torch
from loguru import logger

import ttnn
from models.perf.perf_utils import prep_perf_report
from models.common.utility_functions import run_for_wormhole_b0, is_blackhole
from models.experimental.yolo26.runner.performant_runner import YOLO26PerformantRunner
from models.experimental.yolo26.common import YOLO26_L1_SMALL_SIZE


def get_expected_times(variant, input_size, batch_size):
    """Get expected performance times."""
    expectations = {
        ("yolo26n", 640, 1): (120, 0.015),
        ("yolo26n", 320, 1): (100, 0.008),
    }
    return expectations.get((variant, input_size, batch_size), (120, 0.020))


def run_e2e_performant(device, batch_size, input_size, variant, act_dtype=ttnn.bfloat8_b, weight_dtype=ttnn.bfloat8_b):
    """Run YOLO26 end-to-end performant test."""
    num_devices = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    total_batch_size = batch_size * num_devices

    logger.info(f"Running YOLO26 E2E performant: variant={variant}, input_size={input_size}, batch={total_batch_size}")

    performant_runner = YOLO26PerformantRunner(device, batch_size, input_size, variant, act_dtype, weight_dtype)
    performant_runner.capture_trace_2cq()

    torch_input_tensor = torch.randn(total_batch_size, 3, input_size, input_size)
    iterations_count = 10
    ttnn.synchronize_device(device)

    t0 = time.time()
    for _ in range(iterations_count):
        _ = performant_runner.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()

    performant_runner.release()

    inference_time_avg = (t1 - t0) / iterations_count
    fps = total_batch_size / inference_time_avg

    logger.info(f"YOLO26 {variant}: inference_time={inference_time_avg*1000:.2f}ms, FPS={fps:.1f}")

    expected_compile_time, expected_inference_time = get_expected_times(variant, input_size, batch_size)
    prep_perf_report(
        model_name=f"models/experimental/yolo26/{variant}",
        batch_size=total_batch_size,
        inference_and_compile_time=inference_time_avg,
        inference_time=inference_time_avg,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=f"input_size={input_size}",
        inference_time_cpu=0.0,
    )

    return inference_time_avg, fps


@pytest.mark.parametrize("batch_size, act_dtype, weight_dtype", [(1, ttnn.bfloat8_b, ttnn.bfloat8_b)])
@pytest.mark.parametrize("input_size", [640])
@pytest.mark.parametrize("variant", ["yolo26n"])
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE, "trace_region_size": 8000000, "num_command_queues": 2}],
    indirect=True,
)
def test_yolo26_e2e_performant(device, batch_size, act_dtype, weight_dtype, input_size, variant):
    """YOLO26 end-to-end performant test."""
    inference_time, fps = run_e2e_performant(device, batch_size, input_size, variant, act_dtype, weight_dtype)
    assert fps > 5.0, f"Performance too low: {fps:.1f} FPS"


@pytest.mark.parametrize("batch_size, act_dtype, weight_dtype", [(1, ttnn.bfloat8_b, ttnn.bfloat8_b)])
@pytest.mark.parametrize("input_size", [640])
@pytest.mark.parametrize("variant", ["yolo26n"])
@pytest.mark.models_performance_bare_metal
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE, "trace_region_size": 8000000, "num_command_queues": 2}],
    indirect=True,
)
def test_yolo26_e2e_performant_ci(device, batch_size, act_dtype, weight_dtype, input_size, variant):
    """YOLO26 end-to-end performant CI test."""
    inference_time, fps = run_e2e_performant(device, batch_size, input_size, variant, act_dtype, weight_dtype)
    expected_fps = 50
    margin = 0.15
    min_fps = expected_fps * (1 - margin)
    assert fps > min_fps, f"CI perf below threshold: {fps:.1f} FPS (expected > {min_fps:.1f})"


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only test")
@pytest.mark.parametrize("batch_size, act_dtype, weight_dtype", [(1, ttnn.bfloat8_b, ttnn.bfloat8_b)])
@pytest.mark.parametrize("input_size", [640])
@pytest.mark.parametrize("variant", ["yolo26n"])
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE, "trace_region_size": 8000000, "num_command_queues": 2}],
    indirect=True,
)
def test_yolo26_e2e_performant_blackhole(device, batch_size, act_dtype, weight_dtype, input_size, variant):
    """YOLO26 end-to-end performant test for Blackhole hardware."""
    inference_time, fps = run_e2e_performant(device, batch_size, input_size, variant, act_dtype, weight_dtype)
    assert fps > 10.0, f"Performance too low: {fps:.1f} FPS"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
