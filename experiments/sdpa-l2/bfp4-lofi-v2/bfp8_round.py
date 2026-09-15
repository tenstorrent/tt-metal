# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Experimental native-group RNE BFP8 pre-rounding, followed by device packing.

Independent of the frozen v1 quantizer and all existing v2 implementations.
Inputs are finite BF16 normals or zero. Every nonzero 16-column group's maximum
must have unbiased exponent in [-120, 110]; this makes both the BFP8 grid and
the FP32 magic-rounding constant normal and finite.
RNE removes ties-away rounding bias; saturation at magnitude 127 remains and
can still cause a small distribution-dependent magnitude/gain bias.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DIRECTORY = HERE / "bfp8_round"
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/bfp8_round/"


def validate_input(x):
    assert x.dtype == torch.bfloat16, "This prototype's exact oracle contract is BF16 input"
    assert x.shape[-1] % 16 == 0
    values = x.float()
    assert bool(torch.isfinite(values).all()), "NaN/Inf are outside the prototype contract"
    magnitude = values.abs()
    assert bool(((magnitude == 0) | (magnitude >= 2.0**-126)).all()), "No BF16 subnormal inputs"
    maximum = magnitude.reshape(-1, 16).amax(-1)
    assert bool(((maximum == 0) | ((maximum >= 2.0**-120) & (maximum < 2.0**111))).all()), (
        "Nonzero native-group maximum exponents must be in [-120, 110]"
    )


def host_rne_bfp8(x):
    """Direct shared-exponent 7-magnitude-bit RNE + saturation, host semantics.

    The integer alignment step matches blockfloat_common.cpp exactly. Keeping
    this separate from the magic-add oracle catches arithmetic/model errors.
    """
    validate_input(x)
    groups = x.float().reshape(-1, 16)
    raw = groups.abs().contiguous().view(torch.int32).long()
    exponent = (raw >> 23) & 255
    shared = exponent.amax(-1, keepdim=True)
    significand = torch.where(exponent == 0, 0, (raw & 0x7FFFFF) | 0x800000)
    aligned = significand >> (shared - exponent).clamp_max(63)
    integer = aligned >> 17
    remainder = aligned & ((1 << 17) - 1)
    halfway = 1 << 16
    increment = (remainder > halfway) | ((remainder == halfway) & ((integer & 1) != 0))
    integer = (integer + increment.long()).clamp_max(127)
    quantized = torch.ldexp(integer.float(), (shared - 133).int())
    return (quantized * groups.sign()).reshape_as(x)


def magic_rne_bfp8(x):
    """FP32 arithmetic oracle for the actual kernel, including all-zero groups."""
    validate_input(x)
    groups = x.float().reshape(-1, 16)
    magnitude = groups.abs()
    maximum = magnitude.amax(-1, keepdim=True)
    exponent = torch.frexp(maximum)[1] - 1
    # The kernel uses biased exponent zero for all-zero groups. Both choices
    # below yield zero; choose ordinary constants to avoid host FTZ dependence.
    exponent = torch.where(maximum == 0, 0, exponent)
    magic = torch.ldexp(torch.ones_like(maximum), exponent + 17)
    cap = torch.ldexp(torch.full_like(maximum, 1.984375), exponent)
    rounded = (magnitude + magic) - magic
    return (torch.minimum(rounded, cap) * groups.sign()).reshape_as(x)


def make_input(length, distribution, seed):
    assert length > 0 and length % 32 == 0
    shape = (1, 1, length, 128)
    generator = torch.Generator().manual_seed(seed)
    if distribution == "normal":
        return torch.randn(shape, generator=generator).bfloat16()
    group_count = length * 128 // 16
    group = torch.arange(group_count, dtype=torch.int64)
    lane = torch.arange(16, dtype=torch.int64)
    if distribution == "zeros":
        return torch.zeros(shape, dtype=torch.bfloat16)
    if distribution == "thresholds":
        # Dense BF16 values around every coarse-grid tie. An anchor keeps the
        # native exponent fixed even for small probes. Its varying column and
        # per-group scale catch parity, row, face, and accidental 32-col mixing.
        values = ((group[:, None] + lane[None, :] * 17) % 256).float() / 128
        values[group, group % 16] = 1.984375
        exponent = (group % 17 - 8).int()
    elif distribution == "wide":
        # All 128 BF16 mantissas, both signs, and native-group exponents
        # throughout the admitted range, mixed with much smaller same-group
        # values. This deliberately does not use the v1 -100 exponent clamp.
        mantissa = ((group[:, None] + lane[None, :] * 19) % 128).float()
        values = 1 + mantissa / 128
        delta = (lane % 7).int()
        values = torch.ldexp(values, -delta[None, :])
        values[group, group % 16] = 1.984375
        exponent = (group % 231 - 120).int()
        # Avoid individual BF16 subnormals in low-exponent groups.
        values = torch.where(
            (exponent[:, None] - delta[None, :] < -126), torch.zeros_like(values), values
        )
        values[group, group % 16] = 1.984375
    else:
        raise ValueError(distribution)
    values = torch.ldexp(values, exponent[:, None])
    sign = torch.where(((group[:, None] + lane[None, :]) & 1) != 0, -1.0, 1.0)
    values *= sign
    # Exact zero groups test max=0 and avoid treating the sign bit as magnitude.
    values[::97] = 0
    return values.reshape(shape).bfloat16()


def build(device, src, ncores=1, batch=4, fp32_dst=False, output_format="b8"):
    """Return (output, invoke, actual_cores); source/output lifetimes stay with caller."""
    import ttnn

    assert src.dtype == ttnn.bfloat16
    assert output_format in ("b8", "bf16")
    assert src.shape[-1] % 32 == 0 and src.shape[-2] % 32 == 0
    assert 1 <= batch <= (4 if fp32_dst else 8)
    assert ncores > 0
    dtype, tile_bytes = (ttnn.bfloat8_b, 1088) if output_format == "b8" else (ttnn.bfloat16, 2048)
    out = ttnn.allocate_tensor_on_device(src.shape, dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    tiles = src.volume() // 1024
    assert tiles % batch == 0, "Tile count must divide the DST batch"
    grid_size = device.compute_with_storage_grid_size()
    ncores = min(ncores, tiles // batch, grid_size.x * grid_size.y)
    coords = [ttnn.CoreCoord(i % grid_size.x, i // grid_size.x) for i in range(ncores)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * batch * size,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=size)],
        )
        for index, fmt, size in ((0, ttnn.bfloat16, 2048), (16, dtype, tile_bytes))
    ]
    reader, writer, compute = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    offset = 0
    for i, core in enumerate(coords):
        count = (tiles // batch // ncores + (i < tiles // batch % ncores)) * batch
        reader[core.x][core.y] = [src.buffer_address(), offset, count]
        writer[core.x][core.y] = [out.buffer_address(), offset, count]
        compute[core.x][core.y] = [count]
        offset += count
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
                compile_time_args=[batch],
                runtime_args=compute,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    fp32_dest_acc_en=fp32_dst,
                    math_approx_mode=False,
                ),
            ),
        ],
    )
    return out, lambda: ttnn.generic_op([src, out], desc), ncores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--fp32-dst", action="store_true")
    parser.add_argument("--output-format", choices=("b8", "bf16"), default="b8")
    parser.add_argument("--distribution", choices=("normal", "thresholds", "wide", "zeros"), default="normal")
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trace-repeats", type=int, default=10)
    parser.add_argument("--host-only", action="store_true")
    args = parser.parse_args()
    assert Path(args.label).name == args.label, "Use a plain fresh label, not a path"
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0
    path = DIRECTORY / (args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    x = make_input(args.length, args.distribution, args.seed)
    expected = host_rne_bfp8(x)
    magic = magic_rne_bfp8(x)
    oracle_mismatch = int((magic != expected).sum())
    print("BFP8_ORACLE_CHECK", oracle_mismatch, expected.numel(), flush=True)
    assert oracle_mismatch == 0
    if args.host_only:
        print("Host-only oracle agreement; no device opened.", flush=True)
        return

    import ttnn

    device = ttnn.open_device(device_id=0, trace_region_size=4194304 if args.iters else 0)
    try:
        src = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
        out, invoke, cores = build(device, src, args.cores, args.batch, args.fp32_dst, args.output_format)
        invoke()
        actual = ttnn.to_torch(out).float()
        mismatch = int((actual != expected).sum())
        print("BFP8_ROUND_CHECK", mismatch, actual.numel(), flush=True)
        if mismatch:
            torch.save(dict(input=x, actual=actual, expected=expected), DIRECTORY / (args.label + ".failure.pt"))
        assert mismatch == 0, f"{mismatch} BFP8 RNE mismatches"
        times = []
        if args.iters:
            trace = ttnn.begin_trace_capture(device, cq_id=0)
            for _ in range(args.trace_repeats):
                invoke()
            ttnn.end_trace_capture(device, trace, cq_id=0)
            try:
                for iteration in range(args.warmup + args.iters):
                    start = time.perf_counter()
                    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                    if iteration >= args.warmup:
                        times.append((time.perf_counter() - start) * 1000 / args.trace_repeats)
            finally:
                ttnn.release_trace(device, trace)
            assert torch.equal(actual, ttnn.to_torch(out).float())
        median = statistics.median(times) if times else None
        output_bytes = x.numel() // 1024 * (1088 if args.output_format == "b8" else 2048)
        byte_count = x.numel() * 2 + output_bytes
        record = dict(
            **vars(args),
            actual_cores=cores,
            mismatch=mismatch,
            oracle_mismatch=oracle_mismatch,
            numel=x.numel(),
            median_ms=median,
            replay_ms=times,
            read_write_GBps=byte_count / (median * 1e6) if median else None,
            quantization_l2_pct=float(100 * (actual.double() - x.double()).norm() / x.double().norm())
            if x.double().norm()
            else 0.0,
            least_squares_quantized_gain=float((actual.double() * x.double()).sum() / x.double().square().sum())
            if x.double().square().sum()
            else 1.0,
            relative_absolute_magnitude_drift=float(actual.double().abs().sum() / x.double().abs().sum() - 1)
            if x.double().abs().sum()
            else 0.0,
            source_sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__).resolve(), *sorted(DIRECTORY.glob("*.cpp"))]
            },
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
