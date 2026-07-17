# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Trace + Dual Command Queue (2CQ) performance test for Depth Anything V2.

Follows the exact 2CQ pattern from the official ViT Wormhole reference:
  models/demos/vision/classification/vit/wormhole/tests/test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py

CQ0 executes the captured trace (compute).
CQ1 overlaps host-to-device input transfer and device-to-host output read.
Events synchronize the two queues to avoid data hazards.
"""

import time

import pytest
import torch
from loguru import logger
from transformers import AutoModelForDepthEstimation

import ttnn
from models.experimental.depth_anything_v2.tt.model_def import TtDepthAnythingV2, custom_preprocessor
from models.perf.perf_utils import prep_perf_report

MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
NUM_WARMUP = 50
NUM_MEASURE = 200
# The bounty (#31286 Stage 1) requires at least 15 FPS at 518x518.
# With trace + 2CQ + fully on-device upsampling (no CPU round-trips),
# this target is achievable on N150.
TARGET_FPS = 15.0


def run_trace_2cq(device, tt_model, num_warmup, num_measure):
    """Full 2CQ trace benchmark following the official ViT reference pattern."""

    # ── Persistent DRAM input buffer ────────────────────────────────────
    pixel_values = torch.randn(1, 3, 518, 518)
    tt_inputs_host = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    sharded_dram_cfg = ttnn.DRAM_MEMORY_CONFIG
    tt_image_dram = tt_inputs_host.to(device, sharded_dram_cfg)

    # Bootstrap events so the first loop iteration has something to wait on
    first_op_event = ttnn.record_event(device, 0)
    read_event = ttnn.record_event(device, 1)

    # ── JIT compile pass (CQ0 compute, CQ1 write) ──────────────────────
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_dram, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)

    model_input = ttnn.to_memory_config(tt_image_dram, ttnn.L1_MEMORY_CONFIG)
    first_op_event = ttnn.record_event(device, 0)
    output = tt_model(model_input)
    output_dram = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.record_event(device, 0)
    ttnn.deallocate(output_dram)  # Free JIT-pass output (not consumed)

    # ── Capture trace ───────────────────────────────────────────────────
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_dram, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)

    model_input = ttnn.to_memory_config(tt_image_dram, ttnn.L1_MEMORY_CONFIG)
    first_op_event = ttnn.record_event(device, 0)

    spec = model_input.spec
    input_trace_addr = model_input.buffer_address()
    output.deallocate(force=True)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    output = tt_model(model_input)
    input_l1 = ttnn.allocate_tensor_on_device(spec, device)
    assert input_trace_addr == input_l1.buffer_address(), "L1 address mismatch — trace will use wrong buffer"
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    output_dram = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)

    # ── Warmup with 2CQ ────────────────────────────────────────────────
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_dram, 1)
    write_event = ttnn.record_event(device, 1)

    for _ in range(num_warmup):
        # CQ0: wait for input write, reshard DRAM→L1, execute trace
        ttnn.wait_for_event(0, write_event)
        input_l1 = ttnn.reshard(tt_image_dram, ttnn.L1_MEMORY_CONFIG, input_l1)
        first_op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)

        output_dram = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        last_op_event = ttnn.record_event(device, 0)

        # CQ1: write next input + read current output (overlapped)
        ttnn.wait_for_event(1, first_op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_dram, 1)
        write_event = ttnn.record_event(device, 1)

        ttnn.wait_for_event(1, last_op_event)
        _ = ttnn.from_device(output_dram, blocking=False, cq_id=1)
        read_event = ttnn.record_event(device, 1)

    ttnn.synchronize_device(device)

    # ── Measurement loop ────────────────────────────────────────────────
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_dram, 1)
    write_event = ttnn.record_event(device, 1)

    t0 = time.perf_counter()
    for _ in range(num_measure):
        ttnn.wait_for_event(0, write_event)
        input_l1 = ttnn.reshard(tt_image_dram, ttnn.L1_MEMORY_CONFIG, input_l1)
        first_op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)

        output_dram = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        last_op_event = ttnn.record_event(device, 0)

        ttnn.wait_for_event(1, first_op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_dram, 1)
        write_event = ttnn.record_event(device, 1)

        ttnn.wait_for_event(1, last_op_event)
        _ = ttnn.from_device(output_dram, blocking=False, cq_id=1)
        read_event = ttnn.record_event(device, 1)

    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0

    ttnn.release_trace(device, trace_id)
    return elapsed / num_measure  # avg inference time per frame


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1 << 25}],  # 32MB
    indirect=True,
)
def test_depth_anything_v2_trace_2cq(device):
    """Trace + 2CQ performance test.

    Captures one execution trace on CQ0, overlaps I/O on CQ1, and replays
    to measure peak throughput with zero dispatch and minimal data transfer
    overhead.  Target: >= 15 FPS (bounty #31286 Stage 1 requirement).
    """
    torch_model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, trust_remote_code=True)
    torch_model.eval()

    parameters = custom_preprocessor(torch_model, "depth_anything_v2")
    tt_model = TtDepthAnythingV2(torch_model.config, parameters, device)

    avg_time = run_trace_2cq(device, tt_model, NUM_WARMUP, NUM_MEASURE)
    fps = 1.0 / avg_time

    logger.info(f"avg inference time : {avg_time * 1000:.1f} ms")
    logger.info(f"FPS                : {fps:.1f}")

    prep_perf_report(
        model_name="depth_anything_v2_large_trace_2cq",
        batch_size=1,
        inference_and_compile_time=avg_time,
        inference_time=avg_time,
        expected_compile_time=0,
        expected_inference_time=1.0 / TARGET_FPS,  # 0.067 s
        comments="trace+2cq BS=1 518x518",
        inference_time_cpu=0,
    )

    assert fps >= TARGET_FPS, (
        f"FPS {fps:.1f} below target {TARGET_FPS}. " f"avg={avg_time * 1000:.1f}ms, expected<{1000 / TARGET_FPS:.0f}ms"
    )
