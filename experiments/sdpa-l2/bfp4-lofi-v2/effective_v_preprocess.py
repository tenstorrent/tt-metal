# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Materialize actual LoFi right-operand values: decoded BF16 -> trunc5 BF16.

This is a measurement/preprocessing primitive, not an alternative V quantizer.
Apply it AFTER decoding the actual packed BFP8 tensor. RNE5 before BFP8 packing
does not guarantee that the packer's shared-exponent rounding leaves SrcA5 bits
unchanged. Ordinary finite BF16 normals and zeros are supported; no subnormals.
The copy/pack path canonicalizes negative zero to positive zero; this has no
effect on the consumed-value mean. Nonzero values retain an exact bit contract.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/effective_v_preprocess/"


def oracle(src):
    assert src.dtype == torch.bfloat16
    value = src.float()
    assert bool(torch.isfinite(value).all())
    assert bool(((value == 0) | (value.abs() >= 2.0**-126)).all()), "BF16 subnormals are not supported"
    # BF16 has seven fraction bits. Keep four plus the implicit leading bit.
    raw = src.contiguous().view(torch.int16)
    masked = raw & -8
    return torch.where((raw & 0x7FFF) == 0, 0, masked).to(torch.int16).view(torch.bfloat16)


def check(actual, src):
    expected = oracle(src)
    actual = actual.bfloat16()
    src_bits = src.contiguous().view(torch.int16)
    actual_bits = actual.contiguous().view(torch.int16)
    raw_expected = src_bits & -8
    raw_diff = raw_expected != actual_bits
    mismatch = int((actual_bits != expected.view(torch.int16)).sum())
    nonzero_mismatch = int((raw_diff & ((src_bits & 0x7FFF) != 0)).sum())
    pairs = torch.stack(
        (
            src_bits[raw_diff].int() & 0xFFFF,
            raw_expected[raw_diff].int() & 0xFFFF,
            actual_bits[raw_diff].int() & 0xFFFF,
        ),
        dim=-1,
    )
    unique, counts = torch.unique(pairs, dim=0, return_counts=True)
    result = dict(
        mismatch=mismatch,
        nonzero_bit_mismatch=nonzero_mismatch,
        numel=actual.numel(),
        signed_zero_canonicalizations=int(((src_bits == -32768) & (actual_bits == 0)).sum()),
        raw_bit_mismatch_count=int(raw_diff.sum()),
        raw_bit_pairs=[
            dict(
                input=f"0x{int(row[0]):04x}",
                raw_expected=f"0x{int(row[1]):04x}",
                actual=f"0x{int(row[2]):04x}",
                count=int(count),
            )
            for row, count in zip(unique[:16], counts[:16])
        ],
        raw_pair_count=unique.shape[0],
        contract="Exact nonzero decoded BF16 bits AND0xfff8 (trunc5 toward zero, NOT RNE); signed zeros canonicalized to+0",
    )
    # Print diagnostic bit triples BEFORE a gate can abort, including any
    # nonzero mismatch. Do not silently treat arbitrary value errors as zero.
    print("EFFECTIVE_V_CHECK", json.dumps(result), flush=True)
    assert mismatch == 0 and nonzero_mismatch == 0, f"Effective V trunc5 bitmask mismatch: {result}"
    return result


def source_files():
    return [
        Path(__file__).resolve(),
        *(HERE / "effective_v_preprocess").glob("*.cpp"),
        HERE / "preprocess/reader.cpp",
        HERE / "preprocess/writer.cpp",
    ]


def build(device, src, ncores=1, batch=4, fp32_dst=False):
    """Return (preallocated BF16 output, recompute callable, actual core count)."""
    assert src.dtype == ttnn.bfloat16 and ncores > 0
    assert len(src.shape) == 4 and src.shape[-1] == 128 and src.shape[-2] % 32 == 0
    assert batch in (1, 2, 4) and (not fp32_dst or batch <= 4)
    tiles = src.volume() // 1024
    assert tiles > 0 and tiles % batch == 0
    out = ttnn.allocate_tensor_on_device(src.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    grid_size = device.compute_with_storage_grid_size()
    ncores = min(ncores, tiles // batch, grid_size.x * grid_size.y)
    coords = [ttnn.CoreCoord(i % grid_size.x, i // grid_size.x) for i in range(ncores)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * batch * 2048,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for index in (0, 16)
    ]
    reader, writer, compute = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    offset = 0
    for i, c in enumerate(coords):
        count = (tiles // batch // ncores + (i < tiles // batch % ncores)) * batch
        reader[c.x][c.y] = [src.buffer_address(), offset, count]
        writer[c.x][c.y] = [out.buffer_address(), offset, count]
        compute[c.x][c.y] = [count]
        offset += count
    assert offset == tiles
    desc = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=[],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "reader.cpp",
                core_ranges=grid,
                compile_time_args=[batch] + ttnn.TensorAccessorArgs(src).get_compile_time_args(),
                runtime_args=reader,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "writer.cpp",
                core_ranges=grid,
                compile_time_args=[batch] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=writer,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "compute.cpp",
                core_ranges=grid,
                compile_time_args=[5, batch],
                runtime_args=compute,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=fp32_dst, math_approx_mode=False
                ),
            ),
        ],
    )

    def invoke():
        ttnn.generic_op([src, out], desc)

    return out, invoke, ncores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--batch", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--fp32-dst", action="store_true")
    parser.add_argument("--distribution", choices=("normal", "finite_bf16", "zeros"), default="normal")
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-repeats", type=int, default=10)
    args = parser.parse_args()
    assert args.length > 0 and args.length % 32 == 0 and args.heads > 0
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0
    assert Path(args.label).name == args.label
    path = HERE / (args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    shape = (1, args.heads, args.length, 128)
    if args.distribution == "normal":
        src = torch.randn(shape, generator=torch.Generator().manual_seed(args.seed)).bfloat16()
    else:
        raw = torch.arange(65536, dtype=torch.int32)
        exponent = raw & 0x7F80
        selected = ((exponent != 0) & (exponent != 0x7F80)) | ((raw & 0x7FFF) == 0)
        values = raw[selected].to(torch.int16).view(torch.bfloat16)
        if args.distribution == "zeros":
            values = torch.tensor([0, -32768], dtype=torch.int16).view(torch.bfloat16)
        count = args.heads * args.length * 128
        src = values.repeat((count + values.numel() - 1) // values.numel())[:count].reshape(shape)
    files = source_files()
    provenance = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    device = ttnn.open_device(device_id=0, trace_region_size=4194304 if args.iters else 0)
    try:
        tensor = ttnn.from_torch(src, device=device, layout=ttnn.TILE_LAYOUT)
        out, invoke, cores = build(device, tensor, args.cores, args.batch, args.fp32_dst)
        invoke()
        actual = ttnn.to_torch(out)
        result = check(actual, src)
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
            assert torch.equal(actual.view(torch.int16), ttnn.to_torch(out).view(torch.int16))
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == provenance[str(p.relative_to(ROOT))] for p in files)
        median = statistics.median(times) if times else None
        record = dict(
            **vars(args),
            **result,
            actual_cores=cores,
            median_ms=median,
            replay_ms=times,
            bytes_read_written=4 * src.numel(),
            read_write_GBps=4 * src.numel() / (median * 1e6) if median else None,
            source_sha256=provenance,
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("EFFECTIVE_V_RESULT", json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
