# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YUNet E2E Performance Test using Trace + 2CQ.

Usage:
    # Default 640x640
    pytest models/experimental/yunet/tests/perf/test_e2e_performant.py -v

    # Run with 320x320
    pytest models/experimental/yunet/tests/perf/test_e2e_performant.py -v --input-size 320
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.device import Arch
from models.experimental.yunet.runner.performant_runner import YunetPerformantRunner
from models.experimental.yunet.common import YUNET_L1_SMALL_SIZE

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


# Expected FPS thresholds
EXPECTED_FPS_BLACKHOLE = {
    320: 250,
    640: 60,
}

EXPECTED_FPS_WORMHOLE = {
    320: 150,
    640: 35,
}


def run_yunet_inference(
    device,
    input_height=640,
    input_width=640,
    num_iterations=100,
    act_dtype=ttnn.bfloat16,
):
    """Run YUNet inference and measure performance."""
    performant_runner = YunetPerformantRunner(
        device,
        input_height=input_height,
        input_width=input_width,
        act_dtype=act_dtype,
    )

    # Create input tensors for multiple iterations (NHWC format)
    input_shape = (1, input_height, input_width, 3)
    torch_input_tensors = [torch.randn(input_shape, dtype=torch.bfloat16) for _ in range(num_iterations)]

    # Warmup
    logger.info("Warmup...")
    _ = performant_runner.run(torch_input_tensors[0])
    ttnn.synchronize_device(device)

    # Benchmark
    logger.info(f"Running {num_iterations} iterations...")
    if use_signpost:
        signpost(header="start")

    t0 = time.perf_counter()
    for i in range(num_iterations):
        _ = performant_runner.run(torch_input_tensors[i])
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()

    if use_signpost:
        signpost(header="stop")

    performant_runner.release()

    total_time = t1 - t0
    inference_time_avg = total_time / num_iterations
    fps = num_iterations / total_time

    logger.info(
        f"Model: YUNet - input_size: {input_height}x{input_width}. "
        f"One inference iteration time: {inference_time_avg*1000:.2f} ms, FPS: {fps:.1f}"
    )

    return fps, inference_time_avg


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YUNET_L1_SMALL_SIZE, "trace_region_size": 6000000, "num_command_queues": 2}],
    indirect=True,
)
def test_yunet_e2e_performant(device, input_size):
    """
    End-to-end performance test for YUNet using Trace + 2CQ runner.

    This test measures the throughput including host-to-device transfers.
    """
    input_height, input_width = input_size

    # Select expected FPS based on device architecture
    is_wormhole = device.arch() == Arch.WORMHOLE_B0
    if is_wormhole:
        expected_fps = EXPECTED_FPS_WORMHOLE.get(input_height, 30)
    else:
        expected_fps = EXPECTED_FPS_BLACKHOLE.get(input_height, 60)

    fps, inference_time_avg = run_yunet_inference(
        device,
        input_height=input_height,
        input_width=input_width,
        num_iterations=100,
        act_dtype=ttnn.bfloat16,
    )

    assert fps > expected_fps, f"YUNet {input_height}x{input_width} FPS {fps:.1f} below expected {expected_fps}"
