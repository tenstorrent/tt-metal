# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in compute-throughput benchmark (TEST_SDPA_RECIPE_COMPUTE_BENCH=1).

One core, one head, D128. Q/K/V tiles stay resident in L1 (no DRAM traffic in steady state), so the
timing is the compute loop's. "legacy_exact" is the legacy streaming loop (compute_streaming.hpp,
BF16 dest, HiFi2) with the exact exponential, as DiT callers configure legacy SDPA; FAST is the same
loop with the approximate exponential. Useful FLOPs = 4 * Q * K * D per (Q chunk, K chunk) block.
Trace wall time; dispatch is amortized over ~1.5e11 FLOPs per replay.
"""

import math
import os
import statistics
import time

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
import hashlib

PRECISIONS = {"A": "FAST", "B": "COMPENSATED", "C": "BALANCED", "D": "ACCURATE"}


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def prepare(inputs, variant):
    if not variant.startswith("E_"):
        return inputs
    dtype = {"E_bf16": ttnn.bfloat16, "E_bfp8": ttnn.bfloat8_b, "E_bfp4": ttnn.bfloat4_b}[variant]
    return [
        ttnn.transformer.prepare_sdpa_input(tensor, is_query=index == 0, dtype=ttnn.bfloat16 if index == 0 else dtype)
        for index, tensor in enumerate(inputs)
    ]

pytestmark = [
    pytest.mark.skipif(os.getenv("TEST_SDPA_RECIPE_COMPUTE_BENCH") != "1", reason="Opt-in compute benchmark"),
    pytest.mark.parametrize("device_params", [{"trace_region_size": 16777216}], indirect=True),
]

D = 128
SEQ_K = 8192
TARGET_FLOPS = 1.5e11
VARIANTS = ["legacy_exact", "A", "B", "C", "D", "E_bf16", "E_bfp8", "E_bfp4"]


def build(device, inputs, variant, q_tiles, k_tiles, jobs, chunks):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    recipe = "A" if variant == "legacy_exact" else variant
    T = ttnn._ttnn.operations.transformer
    descriptor = T._sdpa_recipe_compute_program(
        getattr(ttnn.SDPAPrecision, PRECISIONS.get(recipe, "LOW_PRECISION")),
        inputs[1].dtype,
        grid,
        chunks,
        q_tiles=q_tiles,
        k_tiles=k_tiles,
        d_tiles=D // 32,
    )
    compute = descriptor.kernels[0]
    if variant == "legacy_exact":
        compute.defines = [(k, "0" if k == "EXP_APPROX_MODE" else v) for k, v in compute.defines]
    # Perf research (not for merge): SDPA_KO="SDPA_KO_EXP,..." adds knockout defines to the compute kernel.
    extra = [d for d in os.getenv("SDPA_KO", "").split(",") if d]
    if extra:
        compute.defines = list(compute.defines) + [tuple(d.split("=", 1)) if "=" in d else (d, "1") for d in extra]
    fp32 = compute.config.fp32_dest_acc_en
    output = ttnn.allocate_tensor_on_device(
        inputs[0].shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    reader_args, writer_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader_args[0][0] = [x.buffer_address() for x in inputs]
    writer_args[0][0] = [output.buffer_address()]
    compute_args[0][0] = [jobs]
    reader_cta = [jobs, chunks, 1 if fp32 else 2, q_tiles, k_tiles, D // 32]
    for x in inputs:
        reader_cta += ttnn.TensorAccessorArgs(x).get_compile_time_args()
    compute.runtime_args = compute_args
    prefix = "tests/ttnn/unit_tests/operations/sdpa/kernels/"
    descriptor.kernels = [
        ttnn.KernelDescriptor(
            kernel_source=prefix + "reader_resident_bench.cpp",
            core_ranges=grid,
            compile_time_args=reader_cta,
            runtime_args=reader_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=prefix + "writer_resident_bench.cpp",
            core_ranges=grid,
            compile_time_args=[jobs, q_tiles, D // 32] + ttnn.TensorAccessorArgs(output).get_compile_time_args(),
            runtime_args=writer_args,
            config=ttnn.WriterConfigDescriptor(),
        ),
        compute,
    ]
    return lambda: ttnn.generic_op([*inputs, output], descriptor)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("k_chunk", [128, 256, 384, 512] + ([768, 1024] if os.getenv("SDPA_BENCH_BIGK") else []))
@pytest.mark.parametrize("q_chunk", [128, 192, 256, 320])
def test_sdpa_recipe_compute_throughput(device, q_chunk, k_chunk, variant, record_property):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    only_variant = os.getenv("SDPA_BENCH_VARIANT")
    if only_variant and variant not in only_variant.split(","):
        pytest.skip("variant filtered")
    only = os.getenv("SDPA_BENCH_QK")
    if only and f"{q_chunk}x{k_chunk}" not in only.split(","):
        pytest.skip("geometry filtered")
    torch.manual_seed(0)
    host = [torch.randn(1, 1, rows, D).bfloat16() for rows in (q_chunk, k_chunk, k_chunk)]
    # Perf research: SDPA_BENCH_LOGIT_SCALE=f scales q and k by f (logit std f^2) to force rescale events.
    f = float(os.getenv("SDPA_BENCH_LOGIT_SCALE", "1"))
    if f != 1:
        host = [(host[0].float() * f).bfloat16(), (host[1].float() * f).bfloat16(), host[2]]
    inputs = prepare([ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in host], variant)
    chunks = SEQ_K // k_chunk
    block_flops = 4 * q_chunk * k_chunk * D
    jobs = max(1, math.ceil(TARGET_FLOPS / (block_flops * chunks)))
    profile = os.getenv("SDPA_BENCH_PROFILE") == "1"
    if profile:
        jobs = 1  # one Q chunk: the per-phase zones fit the device profiler buffer
    invoke = build(device, inputs, variant, q_chunk // 32, k_chunk // 32, jobs, chunks)
    if profile:
        ttnn.to_torch(invoke())
        ttnn.ReadDeviceProfiler(device)
        return
    try:
        expected = digest(ttnn.to_torch(invoke()))
    except RuntimeError as error:
        text = str(error)
        if "L1" in text or "clash" in text or "circular buffer" in text.lower() or "bytes <= available" in text:
            record_property("status", "does_not_fit_l1")
            pytest.skip(f"{variant} Q{q_chunk}/K{k_chunk} does not fit L1")
        raise
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    output = invoke()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        samples = []
        for iteration in range(8):
            start = time.perf_counter()
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            if iteration >= 2:
                samples.append((time.perf_counter() - start) * 1000)
        if not os.getenv("SDPA_KO"):
            assert digest(ttnn.to_torch(output)) == expected
        median = statistics.median(samples)
        flops = block_flops * chunks * jobs
        record_property("variant", variant)
        record_property("q_chunk", q_chunk)
        record_property("k_chunk", k_chunk)
        record_property("ms_median", median)
        record_property("tflops_per_core", flops / (median * 1e9))
    finally:
        ttnn.release_trace(device, trace)
