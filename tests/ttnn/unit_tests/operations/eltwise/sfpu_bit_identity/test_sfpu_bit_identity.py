# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive bit identity of Blackhole's FP32-accurate sigmoid / SiLU / exp kernels with their frozen references.

kernels/sfpu_reference.h holds the kernels as they were before the SFPU cycle reductions. For every case, the SFPU
writes every input bit pattern the destination can hold into DEST (all 2**32 FP32 encodings, or all 2**16 BF16 ones:
NaNs, infinities, zeros, subnormals included) with no unpacker or packer conversion, runs the reference and the
production kernel on copies, and compares the results bit for bit on the SFPU (kernels/compute_bit_identity.cpp).
The generator cases check that DEST holds exactly the generated patterns. A BF16 destination's store flushes the
subnormals (which shows the comparison catches differences), so the BF16-path kernels also run in an FP32
destination on the BF16 patterns stored as raw bits (bf16_lreg), where they load every pattern, subnormals included.

SFPU_BENCH=1 instead times each kernel, reference and production, SFPU-only on one core (DEST tiles 0..3, repeated)
with the real-time profiler and prints SFPU_BENCH= lines with cycles per tile at SFPU_BENCH_MHZ (default 1350).
"""

import json
import os
import statistics

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole()

_KERNELS = "tests/ttnn/unit_tests/operations/eltwise/sfpu_bit_identity/kernels"
_RESULT_WORDS = 32

OP_GENERATOR, OP_SIGMOID, OP_SIGMOID_R2, OP_SILU, OP_EXP, OP_EXP_R2 = range(6)
# (name, op, FP32 destination, bf16_lreg). bf16_lreg runs the BF16-destination kernel in an FP32 destination on the
# BF16 patterns, so its inputs include the subnormals a BF16-format SFPSTORE (the native BF16 cases' generator) flushes.
CASES = [
    ("generator_fp32", OP_GENERATOR, True, 0),
    ("generator_bf16", OP_GENERATOR, False, 0),
    ("generator_bf16_lreg", OP_GENERATOR, True, 1),
    ("sigmoid_fp32", OP_SIGMOID, True, 0),
    ("sigmoid_fp32_r2", OP_SIGMOID_R2, True, 0),
    ("silu_fp32", OP_SILU, True, 0),
    ("exp_fp32", OP_EXP, True, 0),
    ("exp_fp32_r2", OP_EXP_R2, True, 0),
    ("sigmoid_bf16", OP_SIGMOID, False, 0),
    ("silu_bf16", OP_SILU, False, 0),
    ("sigmoid_bf16_lreg", OP_SIGMOID, True, 1),
    ("silu_bf16_lreg", OP_SILU, True, 1),
]


def _program(cores, core_list, op, fp32, out, ranges, bench_reps=0, bench_variant=0, bf16_lreg=0):
    flag_fmt, flag_page = (ttnn.float32, 4096) if fp32 else (ttnn.bfloat16, 2048)
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * flag_page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=flag_fmt, page_size=flag_page)],
        ),
        ttnn.CBDescriptor(
            total_size=4 * _RESULT_WORDS,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=17, data_format=ttnn.uint32, page_size=4 * _RESULT_WORDS)
            ],
        ),
    ]
    compute_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    for page, (core, (first, count)) in enumerate(zip(core_list, ranges)):
        compute_rt[core.x][core.y] = [first, count]
        writer_rt[core.x][core.y] = [out.buffer_address(), first, count, page]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/compute_bit_identity.cpp",
            core_ranges=cores,
            compile_time_args=[op, bench_reps, bench_variant, bf16_lreg],
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32, math_approx_mode=False
            ),
        )
    ]
    if not bench_reps:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{_KERNELS}/writer_bit_identity.cpp",
                core_ranges=cores,
                compile_time_args=[int(fp32), *ttnn.TensorAccessorArgs(out).get_compile_time_args()],
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            )
        )
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def _io(device, rows):
    return [
        ttnn.from_torch(
            torch.zeros(rows, _RESULT_WORDS, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(2)
    ]


@pytest.mark.skipif(os.environ.get("SFPU_BENCH") == "1", reason="SFPU_BENCH runs the cycle benchmark instead")
@pytest.mark.parametrize("name,op,fp32,bf16_lreg", CASES, ids=[c[0] for c in CASES])
def test_sfpu_bit_identity(device, name, op, fp32, bf16_lreg):
    grid = device.compute_with_storage_grid_size()
    core_list = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    bf16_inputs = bf16_lreg or not fp32
    total = 2**16 // 1024 if bf16_inputs else 2**32 // 1024
    per, extra = divmod(total, len(core_list))
    ranges, first = [], 0
    for i in range(len(core_list)):
        count = per + (i < extra)
        ranges.append((first, count))
        first += count
    io = _io(device, len(core_list))
    ttnn.generic_op(io, _program(cores, core_list, op, fp32, io[1], ranges, bf16_lreg=bf16_lreg))
    result = ttnn.to_torch(io[1]).to(torch.int64) & 0xFFFFFFFF
    checked = int(result[:, 1].sum()) * 1024
    differences = int(result[:, 0].sum())
    shift = 16 if bf16_inputs else 0
    examples = [int(v) << shift for row in result for v in row[2 : 2 + min(int(row[0]), _RESULT_WORDS - 2)]][:16]
    print(
        f"SFPU_BIT_IDENTITY {name}: {checked} inputs ({'all BF16' if bf16_inputs else 'all FP32'} encodings), "
        f"{differences} differ" + (f"; first inputs: {[hex(v) for v in examples]}" if examples else "")
    )
    assert checked == total * 1024
    if op == OP_GENERATOR and not fp32:
        # The BF16-format store flushes the 2 * 127 subnormals, so these reach the native BF16 cases as zeros (the
        # bf16_lreg cases cover them); this case is also the comparison's positive control.
        assert differences == 2 * (2**7 - 1)
        assert all((v >> 23) & 0xFF == 0 and v & 0x7FFFFF for v in examples)
    else:
        assert differences == 0


_BENCH = [
    ("sigmoid_fp32", OP_SIGMOID, True),
    ("sigmoid_fp32_r2", OP_SIGMOID_R2, True),
    ("silu_fp32", OP_SILU, True),
    ("exp_fp32", OP_EXP, True),
    ("exp_fp32_r2", OP_EXP_R2, True),
    ("sigmoid_bf16", OP_SIGMOID, False),
    ("silu_bf16", OP_SILU, False),
]


@pytest.mark.skipif(os.environ.get("SFPU_BENCH") != "1", reason="set SFPU_BENCH=1 for the cycle benchmark")
def test_sfpu_cycles(device):
    from tests.ttnn.profiling.realtime_profiler_utils import (
        profile_realtime_program_merged,
        require_realtime_profiler,
    )

    require_realtime_profiler("the SFPU cycle benchmark")
    mhz = float(os.environ.get("SFPU_BENCH_MHZ", "1350"))
    runs = int(os.environ.get("SFPU_BENCH_RUNS", "5"))
    lo, hi = 64, 1088  # reps of 4 tiles; the difference cancels the program's fixed cost
    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    io = _io(device, 1)
    for name, op, fp32 in _BENCH:
        for variant in (0, 1, 0, 1):  # interleaved: reference, production, reference, production
            times = {}
            for reps in (lo, hi):
                program = _program(cores, [core], op, fp32, io[1], [(0, 0)], bench_reps=reps, bench_variant=variant)
                ttnn.generic_op(io, program)  # compile
                samples = []
                for _ in range(runs):
                    _, programs = profile_realtime_program_merged(device, lambda: ttnn.generic_op(io, program))
                    samples.append(max(p["duration_ns"] for p in programs.values()))
                times[reps] = statistics.median(samples)
            tiles = 4 * (hi - lo)
            entry = {
                "kernel": name,
                "variant": "production" if variant else "reference",
                "ns_per_tile": (times[hi] - times[lo]) / tiles,
                "cycles_per_tile": (times[hi] - times[lo]) / tiles * mhz / 1e3,
            }
            print("SFPU_BENCH=" + json.dumps(entry))
