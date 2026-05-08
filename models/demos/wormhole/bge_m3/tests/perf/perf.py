# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
BGE-M3 embedding performance benchmark.

Two cases: B1/S512 and B32/S512, both with trace capture.
Usage:
    pytest perf.py -k "batch1" -s
    pytest perf.py -k "batch32" -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.wormhole.bge_m3.tt.common import create_tt_model
from models.demos.wormhole.bge_m3.tt.model_config import determine_device_name

# ──────────────────────────────────────────────────────────────────────────────
# Input preparation
# ──────────────────────────────────────────────────────────────────────────────


def prepare_inputs(tokenizer, batch_size, seq_len, pad_token_id):
    """Generate synthetic token inputs on host. Returns dict of torch tensors."""
    input_ids = torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    # Position IDs (RoBERTa-style: pad_token_id + cumsum of non-pad mask)
    mask = (input_ids != pad_token_id).to(torch.int64)
    position_ids = (torch.cumsum(mask, dim=1) * mask + pad_token_id).to(torch.long)

    # Additive attention mask [B, 1, S, S]
    keep = attention_mask.to(torch.bfloat16)
    additive_mask = ((1.0 - keep) * -100000.0).unsqueeze(1).unsqueeze(1).expand(-1, -1, seq_len, -1).contiguous()

    return {
        "input_ids": input_ids,
        "attention_mask": additive_mask,
        "token_type_ids": token_type_ids,
        "position_ids": position_ids,
    }


def to_host_tensors(inputs, mask_dtype):
    """Convert torch tensors to ttnn host tensors (not yet on device)."""

    def ids_to_host(t):
        return ttnn.from_torch(t.int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    return {
        "input_ids": ids_to_host(inputs["input_ids"]),
        "attention_mask": ttnn.from_torch(inputs["attention_mask"], dtype=mask_dtype, layout=ttnn.TILE_LAYOUT),
        "token_type_ids": ids_to_host(inputs["token_type_ids"]),
        "position_ids": ids_to_host(inputs["position_ids"]),
    }


def allocate_device_tensors(host_tensors, mesh_device):
    """Allocate persistent device tensors from host tensors. These are the trace's input slots."""
    dram = ttnn.DRAM_MEMORY_CONFIG
    return {
        "input_ids": host_tensors["input_ids"].to(mesh_device, dram),
        "attention_mask": host_tensors["attention_mask"].to(mesh_device, dram),
        "token_type_ids": host_tensors["token_type_ids"].to(mesh_device, dram),
        "position_ids": host_tensors["position_ids"].to(mesh_device, dram),
    }


def copy_inputs_to_device(host_tensors, device_tensors):
    """Write new host data into the existing device tensor addresses.

    This is how you feed new inputs to a captured trace: the trace reads from
    fixed device memory addresses, so you overwrite those addresses with new
    data before each execute_trace call.
    """
    for key in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
        ttnn.copy_host_to_device_tensor(host_tensors[key], device_tensors[key])


# ──────────────────────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────────────────────

NUM_ITERATIONS = 10
SEQ_LEN = 512


@pytest.mark.parametrize(
    "batch_size",
    [1, 32],
    ids=["batch1", "batch32"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50_000_000, "num_command_queues": 1}],
    indirect=True,
)
def test_embedding_perf(mesh_device, batch_size):
    dtype = ttnn.bfloat8_b
    device_name = determine_device_name(mesh_device)[0]

    # ── Build model ──────────────────────────────────────────────────
    logger.info(f"Building model: B{batch_size} S{SEQ_LEN} {device_name}")
    t0 = time.perf_counter()
    model_args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=batch_size,
        max_seq_len=SEQ_LEN,
        dtype=dtype,
    )
    build_time = time.perf_counter() - t0
    logger.info(f"Model built in {build_time:.1f}s")

    # ── Prepare inputs ───────────────────────────────────────────────
    mask_dtype = dtype if batch_size in (1, 32) else ttnn.bfloat16
    host_inputs = prepare_inputs(model_args.tokenizer, batch_size, SEQ_LEN, model_args.pad_token_id)
    host_tensors = to_host_tensors(host_inputs, mask_dtype)
    device_tensors = allocate_device_tensors(host_tensors, mesh_device)

    # ── Warmup (compile) ─────────────────────────────────────────────
    logger.info("Compiling (first forward)...")
    t0 = time.perf_counter()
    out = model.forward(**device_tensors)
    ttnn.synchronize_device(mesh_device)
    compile_time = time.perf_counter() - t0
    ttnn.deallocate(out)
    logger.info(f"Compile: {compile_time:.2f}s")

    # ── Trace capture ────────────────────────────────────────────────
    logger.info("Capturing trace...")
    trace_output = model.capture_trace(**device_tensors, mesh_device=mesh_device, cq_id=0)

    # Trace warmup
    for _ in range(3):
        model.execute_trace(blocking=True)

    # ── Benchmark ────────────────────────────────────────────────────
    # Each iteration:
    #   1. Generate new random inputs on host
    #   2. Copy them into the persistent device tensors (same addresses the trace reads from)
    #   3. Execute the trace (this is the only part we time)
    #
    # copy_host_to_device_tensor overwrites the device memory that the captured
    # trace will read. This is how you feed fresh data to a trace without
    # re-capturing it.
    logger.info(f"Running {NUM_ITERATIONS} trace replay iterations with fresh inputs...")
    times = []
    for i in range(NUM_ITERATIONS):
        # Generate new random inputs each iteration
        new_host_inputs = prepare_inputs(model_args.tokenizer, batch_size, SEQ_LEN, model_args.pad_token_id)
        new_host_tensors = to_host_tensors(new_host_inputs, mask_dtype)

        # Copy new data into the device tensors the trace reads from
        copy_inputs_to_device(new_host_tensors, device_tensors)

        # Time only the trace execution (device compute)
        t0 = time.perf_counter()
        model.execute_trace(blocking=True)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    model.release_trace()

    # ── Results ──────────────────────────────────────────────────────
    times_ms = [t * 1000 for t in times]
    times_ms.sort()
    avg_ms = sum(times_ms) / len(times_ms)
    best_ms = times_ms[0]
    total_tokens = batch_size * SEQ_LEN

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  BGE-M3 Performance  ({device_name})")
    logger.info("=" * 60)
    logger.info(f"  Batch size:           {batch_size}")
    logger.info(f"  Seq length:           {SEQ_LEN}")
    logger.info(f"  Total tokens:         {total_tokens}")
    logger.info(f"  Iterations:           {NUM_ITERATIONS}")
    logger.info("-" * 60)
    logger.info(f"  Model build time:     {build_time:.1f}s")
    logger.info(f"  Compile (1st run):    {compile_time:.2f}s")
    logger.info("-" * 60)
    logger.info(f"  Avg prefill time:     {avg_ms:.1f}ms")
    logger.info(f"  Best prefill time:    {best_ms:.1f}ms")
    logger.info(f"  Avg embeddings/s:     {batch_size / (avg_ms / 1000):.1f}")
    logger.info(f"  Best embeddings/s:    {batch_size / (best_ms / 1000):.1f}")
    logger.info(f"  Avg tokens/s:         {total_tokens / (avg_ms / 1000):.0f}")
    logger.info(f"  Best tokens/s:        {total_tokens / (best_ms / 1000):.0f}")
    logger.info("=" * 60)
