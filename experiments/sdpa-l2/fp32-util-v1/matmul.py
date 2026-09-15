# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""L1-resident matching no-MOP matmul+pack calibration, not an SDPA benchmark.

Same 1x4 output subblock and FP32 DST/output; full HiFi4. QK inputs are BF16;
PV uses FP32 P storage (hardware matmul conversion) and BF16 V. Inputs
are loaded once and the L1 output is overwritten each repetition, then drained
once. QK uses kt=4/transpose; PV uses kt=32. B=ones permits the same FP64
golden across the two tile-transpose modes. No sustained DRAM/SFPU traffic.
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
PREFIX = "experiments/sdpa-l2/fp32-util-v1/"


def run(device, kind, cores, repetitions):
    kt = {"qk": 4, "pv": 32, "exp": 1}[kind]
    torch.manual_seed(1236)
    a = (torch.rand(32, kt * 32) + 0.5).bfloat16()
    if kind == "pv":
        a = a.float()
    elif kind == "exp":
        a.zero_()
    b = torch.ones(kt * 32, 128, dtype=torch.bfloat16)
    if kind == "exp":
        b.fill_(16)
    ta, tb = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in (a, b)]
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(cores), 1, 32, 128]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    reader, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        reader[x][y] = [ta.buffer_address(), tb.buffer_address(), kt]
        writer[x][y] = [out.buffer_address(), 4, i * 4]
    cbs = [
        ttnn.CBDescriptor(
            total_size=n * page,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)],
        )
        for idx, n, page, dtype in [
            (0, kt, 4096 if kind == "pv" else 2048, ttnn.float32 if kind == "pv" else ttnn.bfloat16),
            (1, kt * 4, 2048, ttnn.bfloat16),
            (16, 4, 4096, ttnn.float32),
        ]
    ]
    desc = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=[],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "matmul_reader.cpp",
                core_ranges=grid,
                compile_time_args=ttnn.TensorAccessorArgs(ta).get_compile_time_args()
                + ttnn.TensorAccessorArgs(tb).get_compile_time_args(),
                runtime_args=reader,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
                core_ranges=grid,
                compile_time_args=[16] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=writer,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + ("exp_compute.cpp" if kind == "exp" else "matmul_compute.cpp"),
                defines=[("SDPA_DIAG_EXP_MODE", "4"), ("SDPA_FP32_EXTRA_CONST", "1")] if kind == "exp" else [],
                core_ranges=grid,
                compile_time_args=[kt, repetitions, int(kind == "qk")],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
            ),
        ],
    )

    def invoke():
        ttnn.generic_op([ta, tb, out], desc)

    invoke()
    actual = ttnn.to_torch(out)
    expected = (a.double() @ b.double()).expand_as(actual)
    if kind == "exp":
        fixed_point = 0.0
        # The exp grid has a deliberate common scale, cancelled by SDPA's
        # normalization. This is not a general standalone exp approximation.
        common_scale = 2 ** (-4 * (32512 - 32500.818359375) / 1024)
        for _ in range(100):
            fixed_point = common_scale * math.exp((fixed_point - 16) / math.sqrt(128))
        expected = torch.zeros_like(actual, dtype=torch.float64)
        expected[..., :64] = fixed_point
        expected[..., 64:96] = 16
    torch.testing.assert_close(actual.double(), expected, rtol=0.001, atol=1e-6 if kind == "exp" else 0.001)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    invoke()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    times = []
    try:
        for i in range(50):
            start = time.perf_counter()
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            if i >= 40:
                times.append((time.perf_counter() - start) * 1000)
        assert torch.equal(actual, ttnn.to_torch(out)), "Replay changed calibration output"
    finally:
        ttnn.release_trace(device, trace)
    flops = len(cores) * repetitions * 2 * 32 * 128 * kt * 32
    result = dict(
        kind=kind,
        cores=len(cores),
        kt=kt,
        repetitions=repetitions,
        trace_replay_ms=times,
        median_ms=statistics.median(times),
        tflops=flops / (statistics.median(times) * 1e9),
        full_output_checked=True,
        fidelity="HiFi4",
        fp32_dst=True,
        warning="L1-resident matmul+pack calibration, not theoretical peak or SDPA utilization",
    )
    if kind == "exp":
        del result["tflops"]
        rate = len(cores) * repetitions * 2048 / (result["median_ms"] / 1000)
        result.update(
            logits_per_second=rate,
            projected_256k_arithmetic_ms=10 * 262144**2 / rate * 1000,
            warning="Register-resident fused sub/exp calibration; excludes reload/pack, matmuls, state updates. Not SDPA performance or a universal lower bound.",
        )
    for tensor in (ta, tb, out):
        ttnn.deallocate(tensor)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--kinds", nargs="+", choices=["qk", "pv", "exp"], default=["qk", "pv"])
    args = parser.parse_args()
    path = HERE / (args.label + ".jsonl")
    assert not path.exists()
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        size = device.compute_with_storage_grid_size()
        cores = [(x, y) for x in range(size.x) for y in range(size.y)]
        for kind in args.kinds:
            r = run(device, kind, cores, 8192 if kind == "pv" else 65536)
            with path.open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)
    finally:
        ttnn.close_device(device)
