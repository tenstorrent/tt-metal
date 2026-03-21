# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLO26 Trace 2CQ Performance Tests.

Tests YOLO26 model performance using trace capture and 2 command queues
for optimal throughput with overlapped data transfer and compute.

Usage:
    # Basic test
    pytest models/experimental/yolo26/tests/perf/test_yolo26_trace_2cq.py -v -s

    # Performance CI test
    pytest models/experimental/yolo26/tests/perf/test_yolo26_trace_2cq.py -v -s -m models_performance_bare_metal
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.perf.perf_utils import prep_perf_report
from models.common.utility_functions import profiler, run_for_wormhole_b0, is_blackhole
from models.experimental.yolo26.runner.performant_runner import YOLO26PerformantRunner
from models.experimental.yolo26.common import YOLO26_L1_SMALL_SIZE

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def get_expected_perf(batch_size: int, input_size: int):
    """
    Get expected performance metrics.

    Returns:
        Tuple of (expected_compile_time_sec, expected_inference_time_sec)
    """
    # Performance targets - these should be updated based on actual measurements
    # Format: (batch_size, input_size): (compile_time, inference_time)
    perf_targets = {
        (1, 640): (120.0, 0.015),  # ~66 FPS target for batch 1
        (1, 320): (100.0, 0.008),  # ~125 FPS target for batch 1, smaller input
        (4, 640): (120.0, 0.045),  # ~88 FPS target for batch 4
        (8, 640): (120.0, 0.080),  # ~100 FPS target for batch 8
    }
    return perf_targets.get((batch_size, input_size), (120.0, 0.020))


def run_trace_2cq_model(
    device,
    runner: YOLO26PerformantRunner,
    num_warmup_iterations: int,
    num_measurement_iterations: int,
):
    """
    Run YOLO26 with trace 2CQ and measure performance.

    This follows the standard trace 2CQ pattern:
    1. Capture trace
    2. Warmup iterations
    3. Measurement iterations with profiling
    """
    # Capture trace
    runner.capture_trace_2cq()

    # Initialize read event
    read_event = ttnn.record_event(device, 1)

    # Warmup runs
    logger.info(f"Running {num_warmup_iterations} warmup iterations...")
    outputs = []

    ttnn.wait_for_event(1, runner.op_event)
    ttnn.copy_host_to_device_tensor(runner.tt_inputs_host, runner.tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)

    for _ in range(num_warmup_iterations):
        ttnn.wait_for_event(0, write_event)
        runner.input_tensor = ttnn.reshard(runner.tt_image_res, runner.input_mem_config, runner.input_tensor)
        first_op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, runner.tid, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)

        output_tensor_dram = ttnn.to_memory_config(runner.runner_infra.output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        last_op_event = ttnn.record_event(device, 0)

        ttnn.wait_for_event(1, first_op_event)
        ttnn.copy_host_to_device_tensor(runner.tt_inputs_host, runner.tt_image_res, 1)
        write_event = ttnn.record_event(device, 1)

        ttnn.wait_for_event(1, last_op_event)
        outputs.append(ttnn.from_device(output_tensor_dram, blocking=False, cq_id=1))
        read_event = ttnn.record_event(device, 1)

    ttnn.synchronize_device(device)

    # Measurement runs
    if use_signpost:
        signpost(header="start")

    outputs = []
    ttnn.wait_for_event(1, runner.op_event)
    ttnn.copy_host_to_device_tensor(runner.tt_inputs_host, runner.tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)

    profiler.start("run")
    for _ in range(num_measurement_iterations):
        ttnn.wait_for_event(0, write_event)
        runner.input_tensor = ttnn.reshard(runner.tt_image_res, runner.input_mem_config, runner.input_tensor)
        first_op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, runner.tid, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)

        output_tensor_dram = ttnn.to_memory_config(runner.runner_infra.output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        last_op_event = ttnn.record_event(device, 0)

        ttnn.wait_for_event(1, first_op_event)
        ttnn.copy_host_to_device_tensor(runner.tt_inputs_host, runner.tt_image_res, 1)
        write_event = ttnn.record_event(device, 1)

        ttnn.wait_for_event(1, last_op_event)
        outputs.append(ttnn.from_device(output_tensor_dram, blocking=False, cq_id=1))
        read_event = ttnn.record_event(device, 1)

    ttnn.synchronize_device(device)
    profiler.end("run")

    if use_signpost:
        signpost(header="stop")

    ttnn.ReadDeviceProfiler(device)
    runner.release()


def run_yolo26_trace_2cq_perf(
    device,
    batch_size: int = 1,
    input_size: int = 640,
    variant: str = "yolo26n",
    num_warmup_iterations: int = 10,
    num_measurement_iterations: int = 100,
):
    """
    Run YOLO26 trace 2CQ performance test.

    Args:
        device: TTNN device
        batch_size: Batch size for inference
        input_size: Input image size (640, 320, etc.)
        variant: Model variant (yolo26n, yolo26s, etc.)
        num_warmup_iterations: Number of warmup iterations
        num_measurement_iterations: Number of measurement iterations
    """
    torch.manual_seed(0)
    profiler.clear()

    logger.info(f"Running YOLO26 trace 2CQ perf: batch_size={batch_size}, input_size={input_size}, variant={variant}")

    # Create performant runner
    runner = YOLO26PerformantRunner(
        device,
        batch_size,
        input_size,
        variant,
    )

    ttnn.synchronize_device(device)

    # Run trace 2CQ benchmark
    run_trace_2cq_model(device, runner, num_warmup_iterations, num_measurement_iterations)

    # Calculate performance metrics
    inference_time_avg = profiler.get("run") / num_measurement_iterations
    expected_compile_time, expected_inference_time = get_expected_perf(batch_size, input_size)

    fps = batch_size / inference_time_avg
    logger.info(f"\n{'='*60}")
    logger.info(f"YOLO26 Trace 2CQ Performance Results")
    logger.info(f"{'='*60}")
    logger.info(f"Variant: {variant}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Input size: {input_size}x{input_size}")
    logger.info(f"Warmup iterations: {num_warmup_iterations}")
    logger.info(f"Measurement iterations: {num_measurement_iterations}")
    logger.info(f"Average inference time: {inference_time_avg*1000:.2f} ms")
    logger.info(f"Throughput: {fps:.1f} FPS")
    logger.info(f"Expected inference time: {expected_inference_time*1000:.2f} ms")
    logger.info(f"{'='*60}")

    # Prepare performance report
    prep_perf_report(
        model_name=f"ttnn_yolo26_{variant}_trace_2cq_batch_{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=0,
        inference_time=inference_time_avg,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=f"input_size={input_size}",
        inference_time_cpu=0,
    )

    return inference_time_avg, fps


# =============================================================================
# Test Cases
# =============================================================================


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE, "trace_region_size": 8000000, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("input_size", [640])
@pytest.mark.parametrize("variant", ["yolo26n"])
def test_yolo26_trace_2cq_perf(device, batch_size, input_size, variant):
    """
    Test YOLO26 trace 2CQ performance.

    This is the main functional test for trace 2CQ performance.
    """
    num_warmup_iterations = 10
    num_measurement_iterations = 50

    inference_time_avg, fps = run_yolo26_trace_2cq_perf(
        device,
        batch_size,
        input_size,
        variant,
        num_warmup_iterations,
        num_measurement_iterations,
    )

    # Basic sanity check - should achieve at least 10 FPS
    assert fps > 10.0, f"Performance too low: {fps:.1f} FPS (expected > 10 FPS)"


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE, "trace_region_size": 8000000, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("input_size", [640])
@pytest.mark.parametrize("variant", ["yolo26n"])
def test_yolo26_trace_2cq_perf_ci(device, batch_size, input_size, variant, is_single_card_n300):
    """
    YOLO26 trace 2CQ CI performance test.

    This test is run in CI pipelines and has stricter performance requirements.
    """
    # Adjust expected performance based on hardware
    if is_single_card_n300:
        expected_fps = 50  # Lower expectation for N300
    else:
        expected_fps = 60  # Higher expectation for N150

    num_warmup_iterations = 50
    num_measurement_iterations = 200

    inference_time_avg, fps = run_yolo26_trace_2cq_perf(
        device,
        batch_size,
        input_size,
        variant,
        num_warmup_iterations,
        num_measurement_iterations,
    )

    # Performance assertion with margin
    margin = 0.1  # 10% margin
    min_fps = expected_fps * (1 - margin)
    max_fps = expected_fps * (1 + margin) * 2  # Allow higher performance

    assert fps > min_fps, f"Performance below minimum: {fps:.1f} FPS (expected > {min_fps:.1f} FPS)"
    logger.info(f"Performance check passed: {fps:.1f} FPS (expected ~{expected_fps} FPS)")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE, "trace_region_size": 8000000, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size, input_size",
    [
        (1, 320),
        (1, 640),
    ],
)
@pytest.mark.parametrize("variant", ["yolo26n"])
def test_yolo26_trace_2cq_perf_sweep(device, batch_size, input_size, variant):
    """
    YOLO26 trace 2CQ performance sweep across different configurations.
    """
    num_warmup_iterations = 10
    num_measurement_iterations = 30

    inference_time_avg, fps = run_yolo26_trace_2cq_perf(
        device,
        batch_size,
        input_size,
        variant,
        num_warmup_iterations,
        num_measurement_iterations,
    )

    # Basic sanity check
    assert fps > 5.0, f"Performance too low: {fps:.1f} FPS"


# =============================================================================
# Blackhole Tests
# =============================================================================


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only test")
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE, "trace_region_size": 8000000, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("input_size", [640])
@pytest.mark.parametrize("variant", ["yolo26n"])
def test_yolo26_trace_2cq_perf_blackhole(device, batch_size, input_size, variant):
    """
    YOLO26 trace 2CQ performance test for Blackhole hardware.
    """
    num_warmup_iterations = 10
    num_measurement_iterations = 50

    inference_time_avg, fps = run_yolo26_trace_2cq_perf(
        device,
        batch_size,
        input_size,
        variant,
        num_warmup_iterations,
        num_measurement_iterations,
    )

    # Blackhole should have better performance
    assert fps > 15.0, f"Performance too low: {fps:.1f} FPS"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
