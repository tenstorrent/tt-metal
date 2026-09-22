# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in matched geometry benchmark: TEST_SDPA_RECIPE_PERF=1 pytest this_file.

Resident timing repeats physical Q/K/V tiles without changing the compute kernel
or CB depths. It diagnoses compute throughput, NOT model attention accuracy.
The distinct-input case times the public operation separately. Trace wall time
includes dispatch/synchronization; do not label it a device-profiler duration.
"""

import os
import statistics
import time

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, digest, make_inputs, prepare, run

pytestmark = [
    pytest.mark.skipif(os.getenv("TEST_SDPA_RECIPE_PERF") != "1", reason="Opt-in performance benchmark"),
    pytest.mark.parametrize("device_params", [{"trace_region_size": 16777216}], indirect=True),
]


def resident(device, inputs, variant, jobs=16, chunks=512):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    descriptor = ttnn._ttnn.operations.transformer._sdpa_recipe_compute_program(
        getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION")), inputs[1].dtype, grid, chunks
    )
    compute = descriptor.kernels[0]
    fp32 = compute.config.fp32_dest_acc_en
    output = ttnn.allocate_tensor_on_device(
        inputs[0].shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    reader_args, writer_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader_args[0][0] = [x.buffer_address() for x in inputs]
    writer_args[0][0] = [output.buffer_address()]
    compute_args[0][0] = [jobs]
    reader_cta = [jobs, chunks, 1 if fp32 else 2]
    for x in inputs:
        reader_cta += ttnn.TensorAccessorArgs(x).get_compile_time_args()
    compute.runtime_args = compute_args
    prefix = "tests/ttnn/unit_tests/operations/sdpa/kernels/"
    descriptor.kernels = [
        ttnn.KernelDescriptor(
            kernel_source=prefix + "reader_resident_recipe.cpp",
            core_ranges=grid,
            compile_time_args=reader_cta,
            runtime_args=reader_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=prefix + "writer_resident_recipe.cpp",
            core_ranges=grid,
            compile_time_args=[jobs] + ttnn.TensorAccessorArgs(output).get_compile_time_args(),
            runtime_args=writer_args,
            config=ttnn.WriterConfigDescriptor(),
        ),
        compute,
    ]
    return lambda: ttnn.generic_op([*inputs, output], descriptor)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", ["resident", "distinct", "changed_max", "with_preparation"])
def test_sdpa_recipe_throughput(device, variant, mode, record_property):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = make_inputs(512 if mode == "resident" else 32768, "changed_max" if mode == "changed_max" else "normal")
    inputs = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in host]
    prepared = prepare(inputs, variant)
    invoke = resident(device, prepared, variant) if mode == "resident" else lambda: run(prepared, variant)
    if mode == "with_preparation":
        invoke = lambda: run(prepare(inputs, variant), variant)
    expected = digest(ttnn.to_torch(invoke()))
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    output = invoke()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        samples = []
        for iteration in range(12):
            start = time.perf_counter()
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if iteration >= 3:
                samples.append(elapsed_ms)
        assert digest(ttnn.to_torch(output)) == expected
        median = statistics.median(samples)
        flops = 4 * 256 * 128 * (512 * 512 * 16 if mode == "resident" else 32768)
        record_property("variant", variant)
        record_property("mode", mode)
        record_property("trace_wall_ms_median", median)
        record_property("trace_wall_ms_min", min(samples))
        record_property("trace_wall_ms_max", max(samples))
        record_property("useful_flops", flops)
        record_property("tflops_per_core", flops / (median * 1e9))
        record_property("preparation_in_timing", mode == "with_preparation")
    finally:
        ttnn.release_trace(device, trace)
