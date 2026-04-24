# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Trace + Dual Command Queue (2CQ) performance test for Depth Anything V2.

This test captures a device execution trace on the first run and replays it
on subsequent iterations, using dual command queues for overlapped data
transfer and compute.  This is the standard tt-metal pattern for peak
throughput measurement.

NOTE: This is a skeleton -- full implementation requires hardware validation
      to tune trace_region_size and verify 2CQ compatibility.
"""

import pytest
import torch
from loguru import logger
from transformers import AutoModelForDepthEstimation

import ttnn
from models.experimental.depth_anything_v2.tt.model_def import TtDepthAnythingV2, custom_preprocessor


BATCH_SIZE = 1
MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
NUM_WARMUP = 5
NUM_ITERATIONS = 20


def run_trace_2cq_depth_anything_v2(device):
    """Run inference with trace capture and dual command queues."""

    # 1. Load reference model and convert weights
    torch_model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, trust_remote_code=True)
    torch_model.eval()

    parameters = custom_preprocessor(torch_model, "depth_anything_v2")
    tt_model = TtDepthAnythingV2(torch_model.config, parameters, device)

    # 2. Create input
    torch.manual_seed(42)
    input_shape = (BATCH_SIZE, 3, 518, 518)
    pixel_values = torch.randn(input_shape)
    tt_input = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # 3. Warmup (compile all ops)
    logger.info("Warming up (compiling)...")
    for _ in range(NUM_WARMUP):
        _output = tt_model(tt_input)
    ttnn.synchronize_device(device)
    logger.info("Warmup complete.")

    # 4. Capture trace
    logger.info("Capturing execution trace...")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    _output = tt_model(tt_input)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    logger.info(f"Trace captured (id={trace_id}).")

    # 5. Replay trace for benchmarking
    logger.info(f"Replaying trace {NUM_ITERATIONS} times...")
    ttnn.synchronize_device(device)

    import time

    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - start

    fps = NUM_ITERATIONS / elapsed
    logger.info(f"Trace replay: {NUM_ITERATIONS} iters in {elapsed:.3f}s = {fps:.1f} FPS")

    # 6. Cleanup
    ttnn.release_trace(device, trace_id)
    logger.info("Trace released.")

    return fps


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 23887872}], indirect=True)
def test_depth_anything_v2_trace_2cq(device):
    """Trace + 2CQ performance test.

    Expected to achieve higher FPS than the non-traced perf test due to
    reduced dispatch overhead from trace replay.
    """
    fps = run_trace_2cq_depth_anything_v2(device)
    logger.info(f"Depth Anything V2 trace+2CQ FPS: {fps:.1f}")
    # Target: > 15 FPS with trace replay
    assert fps > 10, f"FPS {fps:.1f} below minimum threshold of 10"
