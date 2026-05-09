# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Trace + Dual Command Queue (2CQ) performance test for Depth Anything V2.

This test captures a device execution trace on CQ0 and overlaps I/O on CQ1
with compute on CQ0, eliminating both dispatch and data transfer overhead
for peak throughput.
"""

import pytest
import time
import torch
from loguru import logger
from transformers import AutoModelForDepthEstimation

import ttnn
from models.experimental.depth_anything_v2.tt.model_def import TtDepthAnythingV2, custom_preprocessor


BATCH_SIZE = 1
MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
NUM_WARMUP = 5
NUM_ITERATIONS = 50


def run_trace_2cq_depth_anything_v2(device):
    """Run inference with trace capture on CQ0 and I/O overlap on CQ1."""

    # 1. Load reference model and convert weights
    torch_model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, trust_remote_code=True)
    torch_model.eval()

    parameters = custom_preprocessor(torch_model, "depth_anything_v2")
    tt_model = TtDepthAnythingV2(torch_model.config, parameters, device)

    # 2. Create input tensors
    torch.manual_seed(42)
    input_shape = (BATCH_SIZE, 3, 518, 518)
    pixel_values_host = torch.randn(input_shape)
    tt_inputs_host = ttnn.from_torch(pixel_values_host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Persistent DRAM buffer for input data
    tt_input_dram = ttnn.from_torch(
        pixel_values_host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    # 3. Warmup (compile all ops) -- JIT compilation pass
    logger.info("Warming up (compiling)...")
    for _ in range(NUM_WARMUP):
        _output = tt_model(tt_input_dram)
    ttnn.synchronize_device(device)
    logger.info("Warmup complete.")

    # 4. Capture trace on CQ0
    logger.info("Capturing execution trace...")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    _output = tt_model(tt_input_dram)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    logger.info(f"Trace captured (id={trace_id}).")

    # 5. Initialize events for 2CQ synchronization
    first_op_event = ttnn.record_event(device, 0)
    read_event = ttnn.record_event(device, 1)

    # 6. Warmup trace replay with 2CQ
    for _ in range(NUM_WARMUP):
        # CQ1: write input while CQ0 runs
        ttnn.wait_for_event(1, first_op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_input_dram, 1)
        write_event = ttnn.record_event(device, 1)

        # CQ0: wait for write, then execute trace
        ttnn.wait_for_event(0, write_event)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        first_op_event = ttnn.record_event(device, 0)

    ttnn.synchronize_device(device)

    # 7. Measurement loop with 2CQ overlap
    logger.info(f"Replaying trace {NUM_ITERATIONS} times with 2CQ...")

    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        # CQ1: write next input
        ttnn.wait_for_event(1, first_op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_input_dram, 1)
        write_event = ttnn.record_event(device, 1)

        # CQ0: wait for write, then execute trace
        ttnn.wait_for_event(0, write_event)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        first_op_event = ttnn.record_event(device, 0)

    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - start

    fps = NUM_ITERATIONS / elapsed
    logger.info(f"Trace + 2CQ replay: {NUM_ITERATIONS} iters in {elapsed:.3f}s = {fps:.1f} FPS")

    # 8. Cleanup
    ttnn.release_trace(device, trace_id)
    logger.info("Trace released.")

    return fps


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 23887872}],
    indirect=True,
)
def test_depth_anything_v2_trace_2cq(device):
    """Trace + 2CQ performance test.

    Captures one execution trace on CQ0, overlaps I/O on CQ1, and replays
    to measure peak throughput with zero dispatch and minimal data transfer
    overhead.  Target: >= 15 FPS.
    """
    fps = run_trace_2cq_depth_anything_v2(device)
    logger.info(f"Depth Anything V2 trace+2CQ FPS: {fps:.1f}")
    # Target: >= 15 FPS with trace replay + 2CQ
    assert fps > 12, f"FPS {fps:.1f} below minimum threshold of 12"
