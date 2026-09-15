# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent QK + score reload/exp pipeline probe, not SDPA performance."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
PREFIX = "experiments/sdpa-l2/fp32-pipeline-v2/"


def run(device, args):
    size = device.compute_with_storage_grid_size()
    cores = [(x, y) for x in range(size.x) for y in range(size.y)][: args.cores]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    torch.manual_seed(1236)
    a = (torch.rand(32, 128) + 0.5).bfloat16()
    b = torch.ones(128, 128, dtype=torch.bfloat16)
    ta, tb = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in (a, b)]
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(cores), 1, 32, 192]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    reader, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        reader[x][y] = [ta.buffer_address(), tb.buffer_address()]
        writer[x][y] = [out.buffer_address(), 6, i * 6]
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
    )
    pd = ttnn._ttnn.program_descriptor
    modes = pd.VectorUnpackToDestMode([pd.UnpackToDestMode.Default] * 64)
    modes[2] = pd.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    cbs = [
        ttnn.CBDescriptor(
            total_size=n * page,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)],
        )
        for idx, n, page, dtype in [
            (0, 4, 2048, ttnn.bfloat16),
            (1, 16, 2048, ttnn.bfloat16),
            (2, 2, 4096, ttnn.float32),
            (16, 6, 4096, ttnn.float32),
        ]
    ]
    defines = [("SDPA_DIAG_EXP_MODE", "4"), ("SDPA_FP32_EXTRA_CONST", "1")]
    if args.serial:
        defines.append(("PIPE_SERIAL", "1"))
    desc = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=[],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "reader.cpp",
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
                kernel_source=PREFIX + "compute.cpp",
                core_ranges=grid,
                compile_time_args=[args.repetitions],
                defines=defines,
                config=config,
            ),
        ],
    )

    def invoke():
        ttnn.generic_op([ta, tb, out], desc)

    invoke()
    actual = ttnn.to_torch(out)
    expected = torch.empty_like(actual, dtype=torch.float64)
    expected[..., :128] = a.double() @ b.double()
    expected[..., 128:] = 2 ** (-4 * (32512 - 32500.818359375) / 1024) * math.exp(-8 / math.sqrt(128))
    print(
        "OUTPUT_CHECK",
        [
            (
                j,
                actual[..., j * 32 : (j + 1) * 32].min().item(),
                actual[..., j * 32 : (j + 1) * 32].max().item(),
                actual[0, 0, :4, j * 32].tolist(),
                expected[0, 0, :4, j * 32].tolist(),
            )
            for j in range(6)
        ],
        flush=True,
    )
    torch.testing.assert_close(actual.double(), expected, rtol=0.001, atol=1e-6)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    invoke()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    times = []
    try:
        for i in range(15):
            start = time.perf_counter()
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            if i >= 10:
                times.append((time.perf_counter() - start) * 1000)
        assert torch.equal(actual, ttnn.to_torch(out))
    finally:
        ttnn.release_trace(device, trace)
    return dict(
        label=args.label,
        serial=args.serial,
        cores=len(cores),
        repetitions=args.repetitions,
        median_ms=statistics.median(times),
        trace_replay_ms=times,
        full_output_checked=True,
        score_value=actual[0, 0, 0, 128].item(),
        warning=__doc__,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--serial", action="store_true")
    args = parser.parse_args()
    path = HERE / (args.label + ".json")
    assert not path.exists()
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        result = run(device, args)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        ttnn.close_device(device)
