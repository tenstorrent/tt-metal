# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
BGE-M3 embedding performance benchmark.

Two runners, both with trace capture:

  * ``test_embedding_perf``      — single-CQ baseline (current default).
    Each iteration:
        copy_host_to_device_tensor() → execute_trace(blocking=True)
    Reports per-iteration wall time (host sync barrier on every iter).

  * ``test_embedding_perf_2cq``  — dual command-queue overlap.
    CQ0 = trace replay (non-blocking); CQ1 = host→device input copy.
    Event handshake (``record_event`` / ``wait_for_event``) lets the next
    iteration's H2D copy run in parallel with the current iteration's
    device compute (YOLOv8 ``performant_runner.py`` / UNet trace pattern).
    Reports amortized ms/iter — the customer-side handler steady-state.

Usage:
    pytest perf.py -k "batch1" -s            # 1-CQ B=1
    pytest perf.py -k "batch32" -s           # 1-CQ B=32
    pytest perf.py -k "2cq and batch1" -s    # 2-CQ B=1
    pytest perf.py -k "2cq and batch32" -s   # 2-CQ B=32
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


# ==============================================================================
# Mesh-aware input builders (data parallel)
# ==============================================================================
#
# The single-device helpers above (`to_host_tensors`, `allocate_device_tensors`,
# `copy_inputs_to_device`) were written before BGE-M3 supported data parallel
# and do not pass a mesh mapper. These helpers replace them when running on a
# multi-device mesh: each chip receives a different shard of the global batch
# instead of a replicated copy.
#
# Layout (DP=N on a 1xN mesh):
#   global batch [B, S]  --(ShardTensorToMesh(dim=0))-->  [B/N, S] per chip
#
# Weights are auto-replicated by `LazyWeight` (mesh_mapper_config=None), so no
# model code needs the mapper.


def to_dp_host_tensors(torch_inputs, mask_dtype, inputs_mesh_mapper):
    """Build sharded host-side ttnn tensors used to refresh device slots each iter.

    Built with ``device=None`` so they remain on host but carry the mesh shard
    layout. ``ttnn.copy_host_to_device_tensor`` then streams each shard onto
    its matching chip without re-running the mapper.
    """
    return {
        "input_ids": ttnn.from_torch(
            torch_inputs["input_ids"].int(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=inputs_mesh_mapper,
        ),
        "attention_mask": ttnn.from_torch(
            torch_inputs["attention_mask"],
            dtype=mask_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=inputs_mesh_mapper,
        ),
        "token_type_ids": ttnn.from_torch(
            torch_inputs["token_type_ids"].int(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=inputs_mesh_mapper,
        ),
        "position_ids": ttnn.from_torch(
            torch_inputs["position_ids"].int(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=inputs_mesh_mapper,
        ),
    }


def allocate_dp_device_tensors(torch_inputs, mesh_device, mask_dtype, inputs_mesh_mapper):
    """Build the four persistent device input tensors, sharded across the mesh.

    `torch_inputs` is the dict returned by `prepare_inputs(...)` already sized
    at the global batch. Each tensor is sent to device once via the given
    mapper; the captured trace then reads from these addresses every iteration.
    """
    dram = ttnn.DRAM_MEMORY_CONFIG
    return {
        "input_ids": ttnn.from_torch(
            torch_inputs["input_ids"].int(),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=inputs_mesh_mapper,
            memory_config=dram,
        ),
        "attention_mask": ttnn.from_torch(
            torch_inputs["attention_mask"],
            device=mesh_device,
            dtype=mask_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=inputs_mesh_mapper,
            memory_config=dram,
        ),
        "token_type_ids": ttnn.from_torch(
            torch_inputs["token_type_ids"].int(),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=inputs_mesh_mapper,
            memory_config=dram,
        ),
        "position_ids": ttnn.from_torch(
            torch_inputs["position_ids"].int(),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=inputs_mesh_mapper,
            memory_config=dram,
        ),
    }


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
    logger.info(f"  Avg prefill time:     {avg_ms:.3f}ms")
    logger.info(f"  Best prefill time:    {best_ms:.3f}ms")
    logger.info(f"  Avg embeddings/s:     {batch_size / (avg_ms / 1000):.1f}")
    logger.info(f"  Best embeddings/s:    {batch_size / (best_ms / 1000):.1f}")
    logger.info(f"  Avg tokens/s:         {total_tokens / (avg_ms / 1000):.0f}")
    logger.info(f"  Best tokens/s:        {total_tokens / (best_ms / 1000):.0f}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Dual command queue (2-CQ) trace replay
# ─────────────────────────────────────────────────────────────────────────────


def _copy_inputs_to_device_cq(host_tensors, device_tensors, cq_id: int) -> None:
    """H2D copy onto a specific command queue (used for the data-CQ in 2-CQ runs)."""
    for key in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
        ttnn.copy_host_to_device_tensor(host_tensors[key], device_tensors[key], cq_id=cq_id)


@pytest.mark.parametrize(
    "batch_size",
    [1, 32],
    ids=["batch1", "batch32"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50_000_000, "num_command_queues": 2}],
    indirect=True,
)
def test_embedding_perf_2cq(mesh_device, batch_size):
    """Dual command-queue trace replay.

    CQ0 runs the captured trace (compute); CQ1 streams the next iteration's
    inputs from host. Event sync makes sure execute waits for its inputs and
    the next H2D waits for execute to finish reading the input slots.

    Steady-state pattern adapted from
    ``models/demos/yolov8s/runner/performant_runner.py`` and the UNet trace
    test, which is the same primitive Yolo BH-GLX uses for DP=32.
    """
    dtype = ttnn.bfloat8_b
    device_name = determine_device_name(mesh_device)[0]

    # ── Build model ─────────────────────────────────────────────────
    logger.info(f"Building model (2-CQ): B{batch_size} S{SEQ_LEN} {device_name}")
    t0 = time.perf_counter()
    model_args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=batch_size,
        max_seq_len=SEQ_LEN,
        dtype=dtype,
    )
    build_time = time.perf_counter() - t0
    logger.info(f"Model built in {build_time:.1f}s")

    # ── Prepare inputs ────────────────────────────────────────────────
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

    # ── Trace capture (on CQ0) ─────────────────────────────────────────
    logger.info("Capturing trace on CQ0...")
    model.capture_trace(**device_tensors, mesh_device=mesh_device, cq_id=0)

    # Trace warmup (single-CQ blocking is fine here — it's untimed)
    for _ in range(3):
        model.execute_trace(blocking=True)
    ttnn.synchronize_device(mesh_device)

    # ── Benchmark (dual-CQ overlap) ────────────────────────────────────────
    # Steady-state pipeline:
    #   CQ0 (compute):   execute_trace  execute_trace  execute_trace ...
    #   CQ1 (data):      H2D(iter i+1)  H2D(iter i+2)  H2D(iter i+3) ...
    #                    ↑ hidden behind iter i's compute on CQ0
    #
    # Events make CQ0 wait for the input H2D to land, and make CQ1 wait
    # for CQ0 to be done reading the input slots before overwriting them.
    logger.info(f"Running {NUM_ITERATIONS} dual-CQ iterations (amortized timing)")

    # Prime CQ1 with the first iteration's inputs
    op_event = ttnn.record_event(mesh_device, 0)
    ttnn.wait_for_event(1, op_event)
    first_host_inputs = prepare_inputs(model_args.tokenizer, batch_size, SEQ_LEN, model_args.pad_token_id)
    first_host_tensors = to_host_tensors(first_host_inputs, mask_dtype)
    _copy_inputs_to_device_cq(first_host_tensors, device_tensors, cq_id=1)
    write_event = ttnn.record_event(mesh_device, 1)

    t_start = time.perf_counter()
    for _ in range(NUM_ITERATIONS - 1):
        # CQ0 dispatches execute_trace, waiting for the current input H2D
        ttnn.wait_for_event(0, write_event)
        op_event = ttnn.record_event(mesh_device, 0)
        model.execute_trace(blocking=False, synchronize=False)

        # CQ1 starts the *next* iteration's H2D in parallel with CQ0's compute
        ttnn.wait_for_event(1, op_event)
        next_host_inputs = prepare_inputs(model_args.tokenizer, batch_size, SEQ_LEN, model_args.pad_token_id)
        next_host_tensors = to_host_tensors(next_host_inputs, mask_dtype)
        _copy_inputs_to_device_cq(next_host_tensors, device_tensors, cq_id=1)
        write_event = ttnn.record_event(mesh_device, 1)

    # Final iteration: dispatch its execute, then sync once
    ttnn.wait_for_event(0, write_event)
    model.execute_trace(blocking=False, synchronize=False)
    ttnn.synchronize_device(mesh_device)
    t_end = time.perf_counter()

    model.release_trace()

    # ── Results (amortized) ─────────────────────────────────────────────
    total_ms = (t_end - t_start) * 1000.0
    amortized_ms = total_ms / NUM_ITERATIONS
    total_tokens = batch_size * SEQ_LEN

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  BGE-M3 Performance — 2 CQ amortized  ({device_name})")
    logger.info("=" * 60)
    logger.info(f"  Batch size:           {batch_size}")
    logger.info(f"  Seq length:           {SEQ_LEN}")
    logger.info(f"  Total tokens:         {total_tokens}")
    logger.info(f"  Iterations:           {NUM_ITERATIONS}")
    logger.info("-" * 60)
    logger.info(f"  Model build time:     {build_time:.1f}s")
    logger.info(f"  Compile (1st run):    {compile_time:.2f}s")
    logger.info("-" * 60)
    logger.info(f"  Total run time:       {total_ms:.3f}ms")
    logger.info(f"  Amortized ms/iter:    {amortized_ms:.3f}ms")
    logger.info(f"  Embeddings/s (amort): {batch_size / (amortized_ms / 1000):.1f}")
    logger.info(f"  Tokens/s (amort):     {total_tokens / (amortized_ms / 1000):.0f}")
    logger.info("=" * 60)


# ==============================================================================
# Data-parallel trace replay with dual command queue
# ==============================================================================
#
# Opens an Nx mesh device (default 1x8 on this machine), shards the global
# batch across chips, then replays the captured encoder trace with the same
# 2-CQ pattern used by ``test_embedding_perf_2cq``. The per-chip device-time
# is the same as single-device B=per_device_batch (modulo dispatch/fabric).
#
#   per_device_batch=1  + 8 chips  =  global batch 8
#   per_device_batch=32 + 8 chips  =  global batch 256
#
# Both global batches are reported, since steady-state amortized throughput
# scales close to linearly with the chip count.


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 8)],
    indirect=True,
    ids=["dp8"],
)
@pytest.mark.parametrize(
    "per_device_batch",
    [1, 32],
    ids=["perdev1_global8", "perdev32_global256"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50_000_000, "num_command_queues": 2}],
    indirect=True,
)
def test_embedding_perf_dp_2cq(mesh_device, per_device_batch):
    """Data parallel + dual command queue trace replay.

    Global batch is sharded along dim 0 across the mesh (each chip handles
    ``per_device_batch`` examples). Weights are auto-replicated by
    ``LazyWeight``. CQ0 runs the trace, CQ1 streams the next iteration's
    sharded inputs from host — same pattern as
    ``models/demos/yolov8s/runner/performant_runner.py``.
    """
    from models.demos.utils.common_demo_utils import get_mesh_mappers

    dtype = ttnn.bfloat8_b
    device_name = determine_device_name(mesh_device)[0]
    num_devices = mesh_device.get_num_devices()
    global_batch = per_device_batch * num_devices
    mask_dtype = dtype if per_device_batch in (1, 32) else ttnn.bfloat16

    inputs_mesh_mapper, _weights_mesh_mapper, _output_mesh_composer = get_mesh_mappers(mesh_device)
    assert inputs_mesh_mapper is not None, "test_embedding_perf_dp_2cq requires a multi-device mesh; got num_devices=1"

    # ── Build model (per-device view) ────────────────────────────────────────────
    # max_batch_size on the model is per-device; each chip sees `per_device_batch`
    # samples after the input shard. Compute kernel configs are tuned per-shape
    # at this value (see optimizations.py:Optimizations.build).
    logger.info(
        f"Building model (DP+2CQ): per_device_batch={per_device_batch} "
        f"global_batch={global_batch} num_devices={num_devices} {device_name}"
    )
    t0 = time.perf_counter()
    model_args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=per_device_batch,
        max_seq_len=SEQ_LEN,
        dtype=dtype,
    )
    build_time = time.perf_counter() - t0
    logger.info(f"Model built in {build_time:.1f}s")

    # ── Prepare inputs (global batch on host, sharded to device) ────────────────────
    host_inputs = prepare_inputs(model_args.tokenizer, global_batch, SEQ_LEN, model_args.pad_token_id)
    device_tensors = allocate_dp_device_tensors(host_inputs, mesh_device, mask_dtype, inputs_mesh_mapper)

    # ── Warmup (compile) ────────────────────────────────────────────────
    logger.info("Compiling (first forward)...")
    t0 = time.perf_counter()
    out = model.forward(**device_tensors)
    ttnn.synchronize_device(mesh_device)
    compile_time = time.perf_counter() - t0
    ttnn.deallocate(out)
    logger.info(f"Compile: {compile_time:.2f}s")

    # ── Trace capture (on CQ0) ────────────────────────────────────────────────
    logger.info("Capturing trace on CQ0...")
    model.capture_trace(**device_tensors, mesh_device=mesh_device, cq_id=0)
    for _ in range(3):
        model.execute_trace(blocking=True)
    ttnn.synchronize_device(mesh_device)

    # =====================================================================
    # PHASE 1 — 2-CQ amortized run (steady-state throughput)
    # =====================================================================
    # CQ0 = compute (trace); CQ1 = H2D streaming. The whole NUM_ITERATIONS
    # loop is wrapped in a single t_start/t_end so the overlap that 2-CQ
    # provides is reflected in the amortized number. Anything that runs in
    # parallel between CQ0 and CQ1 is "free" in this measurement.
    logger.info(f"Phase 1: {NUM_ITERATIONS} 2-CQ amortized iterations")

    # Prime CQ1 with the first iteration's sharded inputs
    op_event = ttnn.record_event(mesh_device, 0)
    ttnn.wait_for_event(1, op_event)
    first_host = prepare_inputs(model_args.tokenizer, global_batch, SEQ_LEN, model_args.pad_token_id)
    first_host_tensors = to_dp_host_tensors(first_host, mask_dtype, inputs_mesh_mapper)
    _copy_inputs_to_device_cq(first_host_tensors, device_tensors, cq_id=1)
    write_event = ttnn.record_event(mesh_device, 1)

    t_start = time.perf_counter()
    for _ in range(NUM_ITERATIONS - 1):
        ttnn.wait_for_event(0, write_event)
        op_event = ttnn.record_event(mesh_device, 0)
        model.execute_trace(blocking=False, synchronize=False)

        ttnn.wait_for_event(1, op_event)
        next_host = prepare_inputs(model_args.tokenizer, global_batch, SEQ_LEN, model_args.pad_token_id)
        next_host_tensors = to_dp_host_tensors(next_host, mask_dtype, inputs_mesh_mapper)
        _copy_inputs_to_device_cq(next_host_tensors, device_tensors, cq_id=1)
        write_event = ttnn.record_event(mesh_device, 1)

    # Final iteration: dispatch its execute, then sync once
    ttnn.wait_for_event(0, write_event)
    model.execute_trace(blocking=False, synchronize=False)
    ttnn.synchronize_device(mesh_device)
    t_end = time.perf_counter()

    overlap_total_ms = (t_end - t_start) * 1000.0
    overlap_amortized_ms = overlap_total_ms / NUM_ITERATIONS

    # =====================================================================
    # PHASE 2 — split-timing diagnostic (sequential, no overlap)
    # =====================================================================
    # Same captured trace, but each section (pre / H2D / compute / D2H) is
    # timed in isolation with a sync barrier between them. This is NOT what
    # the customer would deploy — it is a *diagnostic* that attributes the
    # per-iter cost to each bucket so you can see where the 2-CQ overlap
    # actually helps and where it can't.
    #
    # Layout per iteration:
    #     1. PRE       — host tokenize + build host-side sharded ttnn tensors
    #     2. H2D       — 4x copy_host_to_device_tensor on CQ0, then sync
    #     3. COMPUTE   — execute_trace(blocking=True) on CQ0 (= dispatch + run + sync)
    #     4. to_torch   — explicit synchronize_device call (this benchmark
    #                    does not read output; sync is the minimum barrier a
    #                    real `ttnn.to_torch(..., mesh_composer=...)` D2H
    #                    pipeline would hit)
    #
    # Use cq_id=0 for both H2D and compute so each step is fully serialized
    # by the C++ command queue.
    logger.info(f"Phase 2: {NUM_ITERATIONS} split-timing diagnostic iterations")

    pre_ms: list[float] = []
    h2d_ms: list[float] = []
    compute_ms: list[float] = []
    to_torch_ms: list[float] = []
    iter_ms: list[float] = []

    for _ in range(NUM_ITERATIONS):
        t_iter = time.perf_counter()

        # 1. Host-side input prep (tokenizer + host tensor builds)
        t0 = time.perf_counter()
        host_inputs = prepare_inputs(model_args.tokenizer, global_batch, SEQ_LEN, model_args.pad_token_id)
        host_tensors = to_dp_host_tensors(host_inputs, mask_dtype, inputs_mesh_mapper)
        pre_ms.append((time.perf_counter() - t0) * 1000.0)

        # 2. H2D (sharded inputs to mesh, sync to ensure landing)
        t0 = time.perf_counter()
        for key in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
            ttnn.copy_host_to_device_tensor(host_tensors[key], device_tensors[key], cq_id=0)
        ttnn.synchronize_device(mesh_device)
        h2d_ms.append((time.perf_counter() - t0) * 1000.0)

        # 3. Compute (blocking execute_trace = dispatch + run + sync)
        t0 = time.perf_counter()
        model.execute_trace(blocking=True)
        compute_ms.append((time.perf_counter() - t0) * 1000.0)

        # 4. to_torch (no readback in this test; pure sync barrier acts as a
        #    placeholder for ttnn.to_torch(..., mesh_composer=...) D2H cost)
        t0 = time.perf_counter()
        ttnn.synchronize_device(mesh_device)
        to_torch_ms.append((time.perf_counter() - t0) * 1000.0)

        iter_ms.append((time.perf_counter() - t_iter) * 1000.0)

    model.release_trace()

    # =====================================================================
    # Results
    # =====================================================================
    def _stats(label: str, samples: list[float]) -> str:
        sorted_s = sorted(samples)
        n = len(sorted_s)
        avg = sum(sorted_s) / n
        return (
            f"  {label:<20} avg={avg:7.3f} ms  "
            f"min={sorted_s[0]:7.3f}  p50={sorted_s[n // 2]:7.3f}  "
            f"max={sorted_s[-1]:7.3f}"
        )

    total_tokens = global_batch * SEQ_LEN

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"  BGE-M3 DP+2CQ Performance  ({device_name})")
    logger.info("=" * 80)
    logger.info(f"  Devices in mesh:        {num_devices}")
    logger.info(f"  Per-device batch:       {per_device_batch}")
    logger.info(f"  Global batch:           {global_batch}")
    logger.info(f"  Seq length:             {SEQ_LEN}")
    logger.info(f"  Tokens / iter (global): {total_tokens}")
    logger.info(f"  Iterations:             {NUM_ITERATIONS}")
    logger.info("-" * 80)
    logger.info(f"  Model build time:       {build_time:.1f}s")
    logger.info(f"  Compile (1st run):      {compile_time:.2f}s")
    logger.info("-" * 80)
    logger.info("  PHASE 1 — 2-CQ amortized (overlap, steady-state)")
    logger.info("-" * 80)
    logger.info(f"  Total run time:         {overlap_total_ms:.3f}ms over {NUM_ITERATIONS} iters")
    logger.info(f"  Amortized ms / iter:    {overlap_amortized_ms:.3f}ms")
    logger.info(f"  Embeddings/s (amort):   {global_batch / (overlap_amortized_ms / 1000):.1f}")
    logger.info(f"  Tokens/s   (amort):     {total_tokens / (overlap_amortized_ms / 1000):.0f}")
    logger.info("-" * 80)
    logger.info("  PHASE 2 — split timing (sequential, no overlap; diagnostic only)")
    logger.info("-" * 80)
    logger.info(_stats("pre (host prep)", pre_ms))
    logger.info(_stats("H2D (cq0)", h2d_ms))
    logger.info(_stats("compute (trace)", compute_ms))
    logger.info(_stats("to_torch (sync)", to_torch_ms))
    logger.info(_stats("iter total", iter_ms))
    logger.info("-" * 80)
    seq_sum = (
        sum(pre_ms) / NUM_ITERATIONS
        + sum(h2d_ms) / NUM_ITERATIONS
        + sum(compute_ms) / NUM_ITERATIONS
        + sum(to_torch_ms) / NUM_ITERATIONS
    )
    logger.info(f"  Sequential sum (pre+H2D+compute+to_torch avg):  {seq_sum:.3f}ms / iter")
    logger.info(f"  Overlap savings (sequential − amortized):  {seq_sum - overlap_amortized_ms:.3f}ms / iter")
    logger.info("=" * 80)


# ==============================================================================
# Three-stage breakdown: H2D -> Forward -> D2H (sync, no overlap)
# ==============================================================================
#
# Sequential per-iter timing of the three stages a customer pays in a serving
# pipeline. No 2-CQ overlap, no async tricks — each stage is fully completed
# (synchronize_device) before the next starts. This is what a single-stream
# customer would measure with naive code.
#
# D2H uses the optimized stack:
#   untilize_with_unpadding (device, multicore)
#   -> on-device copy into a persistent DRAM staging slot
#   -> copy_device_to_host_tensor into a pre-allocated host_staging
#   -> host_staging.batch_to_torch(dest, physical=True, n_threads=0)
#
# H2D uses the existing copy_host_to_device_tensor pattern: pre-build the
# 4 input ttnn tensors once on host, then per-iter overwrite the 4 device
# input slots in place (single sync at end).


def _allocate_d2h_stack(out_dev: ttnn.Tensor, mesh_device, global_batch: int, hidden: int):
    """One-time setup for the optimized D2H path.

    Returns (dram_staging, dest_torch). With ttnn.copy_device_to_torch the
    intermediate ttnn host_staging tensor is no longer needed -- the device
    DMA writes directly into dest_torch.
    """
    b, _, s, d = out_dev.shape
    sample_rm = ttnn.untilize_with_unpadding(out_dev, output_tensor_end=(b - 1, 0, s - 1, d - 1), use_multicore=True)
    dram_staging = ttnn.clone(sample_rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(sample_rm)
    dest_torch = torch.empty((global_batch, 1, s, hidden), dtype=torch.bfloat16)
    return dram_staging, dest_torch


def _d2h_step_optimized(out_dev, mesh_device, dram_staging, dest_torch):
    """Apply the optimized D2H fast path. Returns the dest torch tensor.

    Pipeline:
        1. untilize_with_unpadding on device  -> bf16 ROW_MAJOR
        2. on-device copy into persistent DRAM staging slot
        3. ttnn.copy_device_to_torch: PCIe DMA + 1 memcpy directly to dest_torch
    """
    b, _, s, d = out_dev.shape
    out_rm = ttnn.untilize_with_unpadding(out_dev, output_tensor_end=(b - 1, 0, s - 1, d - 1), use_multicore=True)
    ttnn.copy(out_rm, dram_staging)
    ttnn.deallocate(out_rm)
    ttnn.copy_device_to_torch(dram_staging, dest_torch)
    return dest_torch


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
def test_embedding_perf_h2d_forward_d2h(mesh_device, batch_size):
    """Per-iteration H2D -> Forward -> D2H sequential breakdown.

    Customer-facing perf report. No CQ-overlap. Pure sequential cost of
    each stage in a single-stream serving pipeline. Uses the optimized
    D2H stack (untilize_with_unpadding + copy_device_to_host_tensor +
    batch_to_torch(n_threads=0)).
    """
    dtype = ttnn.bfloat8_b
    device_name = determine_device_name(mesh_device)[0]
    hidden = 1024  # BGE-M3 hidden dim

    # ── Build model ──────────────────────────────────────────────────
    logger.info(f"Building model (H2D->Forward->D2H): B{batch_size} S{SEQ_LEN} {device_name}")
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

    # ── Compile + Trace ──────────────────────────────────────────────
    logger.info("Compiling (first forward)...")
    out = model.forward(**device_tensors)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    logger.info("Capturing trace...")
    trace_out = model.capture_trace(**device_tensors, mesh_device=mesh_device, cq_id=0)

    # Warm trace once before allocating the D2H stack (so trace_out is finalized)
    for _ in range(3):
        model.execute_trace(blocking=True)

    # ── Set up the optimized D2H stack ──────────────────────────────
    # trace_out has the model output tensor (bf8b TILE DRAM, possibly sharded
    # along dim=0 for DP). Allocate the staging buffers ONCE.
    num_devices = mesh_device.get_num_devices()
    global_batch = batch_size * num_devices  # matches trace_out's logical batch dim
    dram_staging, dest_torch = _allocate_d2h_stack(trace_out, mesh_device, global_batch, hidden)

    # ── Pre-build host inputs for H2D (reused across iters) ──────────
    # We use the SAME host tensors each iter — the H2D cost is what we want
    # to measure, not from_torch host-prep.
    h2d_keys = ("input_ids", "attention_mask", "token_type_ids", "position_ids")

    # ── Warmup the 3-stage pipeline ─────────────────────────────────
    logger.info("Warming up H2D + forward + D2H pipeline...")
    for _ in range(3):
        for key in h2d_keys:
            ttnn.copy_host_to_device_tensor(host_tensors[key], device_tensors[key])
        ttnn.synchronize_device(mesh_device)
        model.execute_trace(blocking=True)
        _d2h_step_optimized(trace_out, mesh_device, dram_staging, dest_torch)

    # ── Benchmark ───────────────────────────────────────────────────
    logger.info(f"Measuring {NUM_ITERATIONS} iters of sequential H2D -> Forward -> D2H")
    h2d_times, fwd_times, d2h_times, iter_times = [], [], [], []
    for _ in range(NUM_ITERATIONS):
        t_iter = time.perf_counter()

        # H2D: 4 copies + sync
        t0 = time.perf_counter()
        for key in h2d_keys:
            ttnn.copy_host_to_device_tensor(host_tensors[key], device_tensors[key])
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter()

        # Forward: execute trace (blocking sync inside)
        model.execute_trace(blocking=True)
        t2 = time.perf_counter()

        # D2H: untilize -> copy_device_to_torch (direct PCIe -> torch.Tensor)
        _d2h_step_optimized(trace_out, mesh_device, dram_staging, dest_torch)
        t3 = time.perf_counter()

        h2d_times.append((t1 - t0) * 1000.0)
        fwd_times.append((t2 - t1) * 1000.0)
        d2h_times.append((t3 - t2) * 1000.0)
        iter_times.append((t3 - t_iter) * 1000.0)

    model.release_trace()

    # ── Report ──────────────────────────────────────────────────────
    def _stats(samples):
        s = sorted(samples)
        n = len(s)
        return {
            "avg": sum(s) / n,
            "min": s[0],
            "p50": s[n // 2],
            "max": s[-1],
        }

    h2d_s = _stats(h2d_times)
    fwd_s = _stats(fwd_times)
    d2h_s = _stats(d2h_times)
    iter_s = _stats(iter_times)

    logger.info("")
    logger.info("=" * 90)
    logger.info(
        f"  BGE-M3 H2D -> Forward -> D2H breakdown  |  batch={batch_size}  S={SEQ_LEN}  "
        f"{device_name}  ({num_devices} chip{'s' if num_devices > 1 else ''})"
    )
    logger.info("=" * 90)
    logger.info(
        f"  {'Stage':<10} | {'avg ms':>10} | {'min ms':>10} | {'p50 ms':>10} | {'max ms':>10} | {'% of iter':>10}"
    )
    logger.info("-" * 90)
    iter_avg = iter_s["avg"]
    for name, s in (("H2D", h2d_s), ("Forward", fwd_s), ("D2H", d2h_s)):
        pct = (s["avg"] / iter_avg) * 100.0 if iter_avg > 0 else 0.0
        logger.info(
            f"  {name:<10} | {s['avg']:>10.3f} | {s['min']:>10.3f} | {s['p50']:>10.3f} | {s['max']:>10.3f} | {pct:>9.1f}%"
        )
    logger.info("-" * 90)
    logger.info(
        f"  {'TOTAL':<10} | {iter_s['avg']:>10.3f} | {iter_s['min']:>10.3f} | {iter_s['p50']:>10.3f} | {iter_s['max']:>10.3f} | {100.0:>9.1f}%"
    )
    logger.info("=" * 90)
    logger.info(
        f"  Throughput (sequential, no overlap): {batch_size / (iter_avg / 1000):.1f} embeddings/s ({batch_size * SEQ_LEN / (iter_avg / 1000):.0f} tokens/s)"
    )
    logger.info("=" * 90)


# ==============================================================================
# Data-parallel three-stage breakdown: H2D -> Forward -> D2H (sync, no overlap)
# ==============================================================================
#
# Same single-stream sequential breakdown as `test_embedding_perf_h2d_forward_d2h`,
# but parametrized for multi-chip DP execution. Use this to characterize
# H2D / Forward / D2H costs on Blackhole Galaxy (DP=32) and similar large meshes.
#
# Configurations:
#   dp8  + perdev1   = (1, 8) mesh, per-device batch 1  -> global  8
#   dp8  + perdev32  = (1, 8) mesh, per-device batch 32 -> global  256
#   dp32 + perdev32  = (4, 8) mesh, per-device batch 32 -> global  1024  (Galaxy)
#
# Select a config with `-k`, e.g.
#   pytest perf.py::test_embedding_perf_h2d_forward_d2h_dp -k "dp32"


@pytest.mark.parametrize(
    "mesh_device, per_device_batch",
    [
        ((1, 8), 1),
        ((1, 8), 32),
        ((4, 8), 32),
    ],
    indirect=["mesh_device"],
    ids=["dp8_perdev1_global8", "dp8_perdev32_global256", "dp32_perdev32_global1024"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50_000_000, "num_command_queues": 1}],
    indirect=True,
)
def test_embedding_perf_h2d_forward_d2h_dp(mesh_device, per_device_batch):
    """Per-iteration H2D -> Forward -> D2H breakdown across a data-parallel mesh.

    Customer-facing perf report for sharded execution. Each chip handles
    ``per_device_batch`` samples; the global batch is ``per_device_batch *
    num_devices``. Sequential single-stream timing, no CQ overlap.

    The D2H stack uses ``ttnn.copy_device_to_torch`` to write the full
    ``[global_batch, 1, S, hidden]`` output directly into a pre-allocated
    torch.Tensor, bypassing the host_staging intermediate.
    """
    from models.demos.utils.common_demo_utils import get_mesh_mappers

    dtype = ttnn.bfloat8_b
    device_name = determine_device_name(mesh_device)[0]
    num_devices = mesh_device.get_num_devices()
    global_batch = per_device_batch * num_devices
    mask_dtype = dtype if per_device_batch in (1, 32) else ttnn.bfloat16
    hidden = 1024

    inputs_mesh_mapper, _weights_mesh_mapper, _output_mesh_composer = get_mesh_mappers(mesh_device)
    assert inputs_mesh_mapper is not None, (
        "test_embedding_perf_h2d_forward_d2h_dp requires a multi-device mesh; " f"got num_devices={num_devices}"
    )

    # ── Build model (per-device view) ────────────────────────────────────
    logger.info(
        f"Building model (DP H2D->Forward->D2H): per_device_batch={per_device_batch} "
        f"global_batch={global_batch} num_devices={num_devices} {device_name}"
    )
    t0 = time.perf_counter()
    model_args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=per_device_batch,
        max_seq_len=SEQ_LEN,
        dtype=dtype,
    )
    build_time = time.perf_counter() - t0
    logger.info(f"Model built in {build_time:.1f}s")

    # ── Prepare inputs (global batch, sharded across mesh) ───────────────
    host_inputs = prepare_inputs(model_args.tokenizer, global_batch, SEQ_LEN, model_args.pad_token_id)
    host_tensors = to_dp_host_tensors(host_inputs, mask_dtype, inputs_mesh_mapper)
    device_tensors = allocate_dp_device_tensors(host_inputs, mesh_device, mask_dtype, inputs_mesh_mapper)

    # ── Compile + Trace ──────────────────────────────────────────────────
    logger.info("Compiling (first forward)...")
    out = model.forward(**device_tensors)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    logger.info("Capturing trace...")
    trace_out = model.capture_trace(**device_tensors, mesh_device=mesh_device, cq_id=0)

    # Warm trace once before allocating the D2H stack
    for _ in range(3):
        model.execute_trace(blocking=True)

    # ── Set up the optimized D2H stack ──────────────────────────────────
    # trace_out has logical shape [global_batch, 1, S, hidden] sharded along
    # dim 0 across the mesh. Allocate the staging buffers ONCE.
    dram_staging, dest_torch = _allocate_d2h_stack(trace_out, mesh_device, global_batch, hidden)

    # ── H2D handle keys ─────────────────────────────────────────────────
    h2d_keys = ("input_ids", "attention_mask", "token_type_ids", "position_ids")

    # ── Warmup the 3-stage pipeline ─────────────────────────────────────
    logger.info("Warming up H2D + forward + D2H pipeline...")
    for _ in range(3):
        for key in h2d_keys:
            ttnn.copy_host_to_device_tensor(host_tensors[key], device_tensors[key])
        ttnn.synchronize_device(mesh_device)
        model.execute_trace(blocking=True)
        _d2h_step_optimized(trace_out, mesh_device, dram_staging, dest_torch)

    # ── Benchmark ───────────────────────────────────────────────────────
    logger.info(f"Measuring {NUM_ITERATIONS} iters of sequential H2D -> Forward -> D2H")
    h2d_times, fwd_times, d2h_times, iter_times = [], [], [], []
    for _ in range(NUM_ITERATIONS):
        t_iter = time.perf_counter()

        t0 = time.perf_counter()
        for key in h2d_keys:
            ttnn.copy_host_to_device_tensor(host_tensors[key], device_tensors[key])
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter()

        model.execute_trace(blocking=True)
        t2 = time.perf_counter()

        _d2h_step_optimized(trace_out, mesh_device, dram_staging, dest_torch)
        t3 = time.perf_counter()

        h2d_times.append((t1 - t0) * 1000.0)
        fwd_times.append((t2 - t1) * 1000.0)
        d2h_times.append((t3 - t2) * 1000.0)
        iter_times.append((t3 - t_iter) * 1000.0)

    model.release_trace()

    # ── Report ──────────────────────────────────────────────────────────
    def _stats(samples):
        s = sorted(samples)
        n = len(s)
        return {
            "avg": sum(s) / n,
            "min": s[0],
            "p50": s[n // 2],
            "max": s[-1],
        }

    h2d_s = _stats(h2d_times)
    fwd_s = _stats(fwd_times)
    d2h_s = _stats(d2h_times)
    iter_s = _stats(iter_times)

    logger.info("")
    logger.info("=" * 100)
    logger.info(
        f"  BGE-M3 H2D -> Forward -> D2H breakdown (DP)  |  per_device_batch={per_device_batch}  "
        f"global_batch={global_batch}  S={SEQ_LEN}  {device_name}  ({num_devices} chips)"
    )
    logger.info("=" * 100)
    logger.info(
        f"  {'Stage':<10} | {'avg ms':>10} | {'min ms':>10} | {'p50 ms':>10} | {'max ms':>10} | {'% of iter':>10}"
    )
    logger.info("-" * 100)
    iter_avg = iter_s["avg"]
    for name, s in (("H2D", h2d_s), ("Forward", fwd_s), ("D2H", d2h_s)):
        pct = (s["avg"] / iter_avg) * 100.0 if iter_avg > 0 else 0.0
        logger.info(
            f"  {name:<10} | {s['avg']:>10.3f} | {s['min']:>10.3f} | {s['p50']:>10.3f} | {s['max']:>10.3f} | {pct:>9.1f}%"
        )
    logger.info("-" * 100)
    logger.info(
        f"  {'TOTAL':<10} | {iter_s['avg']:>10.3f} | {iter_s['min']:>10.3f} | {iter_s['p50']:>10.3f} | {iter_s['max']:>10.3f} | {100.0:>9.1f}%"
    )
    logger.info("=" * 100)
    logger.info(
        f"  Throughput (sequential, no overlap): "
        f"{global_batch / (iter_avg / 1000):.1f} embeddings/s "
        f"({global_batch * SEQ_LEN / (iter_avg / 1000):.0f} tokens/s)"
    )
    logger.info("=" * 100)
