# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""On-device per-value RNE rounding for LoFi Q (7 bits) and K/V (5 bits)."""
import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/preprocess/"
spec = importlib.util.spec_from_file_location("lofi_numeric_model", HERE / "numerics.py")
MODEL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODEL)


def build(device, src, bits=5, output_format="bf16", ncores=1, batch=4, fp32_dst=True, bfp8_pack_precise=False):
    """Return output tensor and a callable; callers retain source/output lifetimes."""
    dtype, tile_bytes = (ttnn.bfloat16, 2048) if output_format == "bf16" else (ttnn.bfloat8_b, 1088)
    out = ttnn.allocate_tensor_on_device(src.shape, dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    tiles = src.volume() // 1024
    assert tiles % batch == 0
    grid_size = device.compute_with_storage_grid_size()
    ncores = min(ncores, tiles // batch, grid_size.x * grid_size.y)
    coords = [ttnn.CoreCoord(i % grid_size.x, i // grid_size.x) for i in range(ncores)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    cbs = [ttnn.CBDescriptor(total_size=2 * batch * size, core_ranges=grid,
                            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=size)])
           for index, fmt, size in ((0, ttnn.bfloat16, 2048), (16, dtype, tile_bytes))]
    reader, writer, compute = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    offset = 0
    for i, c in enumerate(coords):
        count = (tiles // batch // ncores + (i < tiles // batch % ncores)) * batch
        reader[c.x][c.y] = [src.buffer_address(), offset, count]
        writer[c.x][c.y] = [out.buffer_address(), offset, count]
        compute[c.x][c.y] = [count]
        offset += count
    desc = ttnn.ProgramDescriptor(cbs=cbs, semaphores=[], kernels=[
        ttnn.KernelDescriptor(kernel_source=PREFIX + "reader.cpp", core_ranges=grid,
                              compile_time_args=[batch] + ttnn.TensorAccessorArgs(src).get_compile_time_args(),
                              runtime_args=reader, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=PREFIX + "writer.cpp", core_ranges=grid,
                              compile_time_args=[batch] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                              runtime_args=writer, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=PREFIX + "compute.cpp", core_ranges=grid,
                              compile_time_args=[bits, batch], runtime_args=compute,
                              config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi,
                                                                  fp32_dest_acc_en=fp32_dst, math_approx_mode=False,
                                                                  bfp8_pack_precise=bfp8_pack_precise))])
    return out, lambda: ttnn.generic_op([src, out], desc), ncores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--bits", type=int, choices=(5, 7, 8), default=5)
    parser.add_argument("--bfp8-pack-precise", action="store_true")
    parser.add_argument("--output-format", choices=("bf16", "b8"), default="bf16")
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--bf16-dst", action="store_true")
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trace-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--distribution", choices=("normal", "finite_bf16"), default="normal")
    args = parser.parse_args()
    path = HERE / (args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    if args.distribution == "normal":
        x = torch.randn((1, 1, args.length, 128), generator=torch.Generator().manual_seed(args.seed)).bfloat16()
    else:
        # Both signs, all mantissas and ordinary exponents; avoid overflow and subnormals.
        raw = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
        raw = raw[(raw.float().abs() >= 2.0**-120) & (raw.float().abs() < 2.0**120)]
        x = raw.repeat((args.length * 128 + raw.numel() - 1) // raw.numel())[:args.length * 128].reshape(1, 1, args.length, 128)
    expected = MODEL.round_significand(x, args.bits)
    if args.output_format == "b8":
        if not args.bfp8_pack_precise:
            # Generic pack defaults use an E8M6 intermediate before shared
            # exponent conversion. This is identity for already-RNE5/7 values.
            expected = MODEL.round_significand(expected, 7, "rna")
        # The frozen v1 quantizer deliberately clamps exponents below -100.
        # Scale each independent native group into ordinary range before
        # applying it, then undo that exact power-of-two scale.
        groups = expected.reshape(-1, 16)
        exponents = torch.frexp(groups.abs().amax(-1, keepdim=True))[1]
        shifts = torch.where(exponents < -90, -exponents, 0)
        scaled = torch.ldexp(groups.double(), shifts).float()
        expected = torch.ldexp(MODEL.quantize(scaled, 7, "device").double(), -shifts).float().reshape_as(expected)
    device = ttnn.open_device(device_id=0, trace_region_size=4194304 if args.iters else 0)
    try:
        src = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
        out, invoke, cores = build(device, src, args.bits, args.output_format, args.cores, args.batch, not args.bf16_dst,
                                   args.bfp8_pack_precise)
        invoke()
        actual = ttnn.to_torch(out).float()
        mismatch = int((actual != expected).sum())
        print("ROUND_CHECK", mismatch, actual.numel(), flush=True)
        if mismatch:
            torch.save(dict(input=x, actual=actual, expected=expected), HERE / (args.label + ".failure.pt"))
        assert mismatch == 0, f"{mismatch} rounding mismatches"
        times = []
        if args.iters:
            trace = ttnn.begin_trace_capture(device, cq_id=0)
            for _ in range(args.trace_repeats):
                invoke()
            ttnn.end_trace_capture(device, trace, cq_id=0)
            try:
                for i in range(args.warmup + args.iters):
                    start = time.perf_counter()
                    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                    if i >= args.warmup:
                        times.append((time.perf_counter() - start) * 1000 / args.trace_repeats)
            finally:
                ttnn.release_trace(device, trace)
            assert torch.equal(actual, ttnn.to_torch(out).float())
        median = statistics.median(times) if times else None
        byte_count = x.numel() * 2 + x.numel() // 1024 * (2048 if args.output_format == "bf16" else 1088)
        record = dict(**vars(args), actual_cores=cores, mismatch=mismatch, numel=x.numel(),
                      median_ms=median, replay_ms=times, read_write_GBps=byte_count / (median * 1e6) if median else None,
                      source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in [Path(__file__).resolve(), *(HERE / "preprocess").glob("*.cpp")]})
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)
