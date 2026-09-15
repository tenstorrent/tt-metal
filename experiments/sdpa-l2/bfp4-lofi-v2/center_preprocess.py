# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused BF16 K minus BF16 per-head bias, FP32 SFPU centering, then quantization.

Independent prototype. No centered-value BF16 spill. Modes are native BFP4
with direct FP32 RNE pre-rounding, or per-value RNE5 followed by native BFP8.
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
DIRECTORY = HERE / "center_preprocess"
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/center_preprocess/"


def validate_centered(x):
    assert x.dtype == torch.float32 and x.shape[-1] == 128
    assert bool(torch.isfinite(x).all()), "Nonfinite centered values are outside the contract"
    magnitude = x.abs()
    assert bool(((magnitude == 0) | (magnitude >= 2.0**-126)).all()), "No subnormal centered values"
    maximum = magnitude.reshape(-1, 16).amax(-1)
    assert bool(((maximum == 0) | ((maximum >= 2.0**-124) & (maximum < 2.0**107))).all()), (
        "Nonzero centered group maximum exponents must be in [-124,106]"
    )


def center_oracle(src, bias):
    assert src.dtype == bias.dtype == torch.bfloat16
    assert src.ndim == 4 and src.shape[0] == 1 and src.shape[-1] == 128
    assert tuple(bias.shape) == (1, src.shape[1], 32, 128)
    assert torch.equal(bias, bias[:, :, :1, :].expand_as(bias)), "Bias rows must repeat the column means"
    for value in (src, bias):
        value = value.float()
        assert bool(torch.isfinite(value).all())
        assert bool(((value == 0) | (value.abs() >= 2.0**-126)).all()), "No BF16 input/bias subnormals"
    centered = src.float() - bias[:, :, :1, :].float()
    validate_centered(centered)
    return centered


def quantize_oracle(centered, mode):
    validate_centered(centered)
    groups = centered.reshape(-1, 16)
    if mode == "b4_rne":
        magnitude = groups.abs()
        maximum = magnitude.amax(-1, keepdim=True)
        exponent = torch.frexp(maximum)[1] - 1
        exponent = torch.where(maximum == 0, 0, exponent)
        magic = torch.ldexp(torch.ones_like(maximum), exponent + 21)
        cap = torch.ldexp(torch.full_like(maximum, 1.75), exponent)
        quantized = torch.minimum((magnitude + magic) - magic, cap)
        # Independent FP64 scaling/RNE oracle retains all FP32 input bits.
        # The old BF16-only aligned-integer oracle can discard sticky bits.
        integer = torch.round(torch.ldexp(magnitude.double(), 2 - exponent)).clamp_max(7)
        exact = torch.ldexp(integer, exponent - 2).float()
        assert torch.equal(quantized, exact), "FP32 magic and direct FP64 grid disagree"
        return (quantized * groups.sign()).reshape_as(centered)
    assert mode == "b8_rne5"
    raw = groups.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    rounded = ((raw + 0x3FFFF + ((raw >> 19) & 1)) & 0xFFF80000).int().view(torch.float32)
    raw = rounded.abs().contiguous().view(torch.int32).long()
    exponent = (raw >> 23) & 255
    shared = exponent.amax(-1, keepdim=True)
    significand = torch.where(exponent == 0, 0, ((raw >> 16) & 127) | 128)
    shift = shared - exponent + 1
    safe_shift = shift.clamp_max(9)
    integer = (significand + (1 << (safe_shift - 1))) >> safe_shift
    integer = torch.where(shift > 8, 0, integer).clamp_max(127)
    return (torch.ldexp(integer.float(), (shared - 133).int()) * rounded.sign()).reshape_as(centered)


def make_input(length, heads, distribution, seed):
    assert length > 0 and length % 32 == 0 and heads > 0
    shape = (1, heads, length, 128)
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(shape, generator=generator)
    head = torch.arange(heads).reshape(1, heads, 1, 1)
    column = torch.arange(128).reshape(1, 1, 1, 128)
    if distribution == "normal":
        src = values.bfloat16()
    elif distribution == "common_k":
        common = (head + 1) * 4.0 + (column % 13) * 0.125
        src = (values + common).bfloat16()
    elif distribution == "wide":
        exponent = ((head * 17 + column * 7) % 161 - 80).int()
        src = torch.ldexp(1.5 + values * 0.25, exponent).bfloat16()
    elif distribution == "zeros":
        src = torch.zeros(shape, dtype=torch.bfloat16)
    elif distribution == "thresholds":
        # Exact BF16 examples where FP32 centering and a BF16-centered spill
        # choose different RNE bins. Distinct heads/columns also test caching.
        row = torch.arange(length).reshape(1, 1, length, 1)
        src = (1.03125 + ((row + column + head) % 2) * 0.09375).expand(shape).bfloat16()
        bias = torch.full((1, heads, 32, 128), -(2.0**-10), dtype=torch.bfloat16)
        return src, bias
    else:
        raise ValueError(distribution)
    # Mean generation is explicitly outside this device prototype/timing.
    mean = src.float().mean(dim=2, keepdim=True).bfloat16()
    bias = mean.expand(1, heads, 32, 128).contiguous()
    return src, bias


def core_segments(tiles, ncores):
    assert tiles % 4 == 0 and 0 < ncores <= tiles // 4
    offset, segments = 0, []
    for i in range(ncores):
        count = (tiles // 4 // ncores + (i < tiles // 4 % ncores)) * 4
        segments.append((offset, count))
        offset += count
    assert offset == tiles
    return segments


def build(device, src, bias, ncores=1, mode="b8_rne5"):
    """Return (output, invoke, actual_cores); caller enforces the value contract.

    src: BF16 [1,H,N,128], bias: BF16 [1,H,32,128], TILE/DRAM tensors.
    BF16 DST, fixed batch4; centered intermediates remain FP32 SFPU values.
    """
    import ttnn

    assert mode in ("b8_rne5", "b4_rne") and ncores > 0
    assert src.dtype == bias.dtype == ttnn.bfloat16
    assert len(src.shape) == 4 and src.shape[0] == 1 and src.shape[-1] == 128
    assert src.shape[2] > 0 and src.shape[2] % 32 == 0
    assert tuple(bias.shape) == (1, src.shape[1], 32, 128)
    fmt, out_bytes = (ttnn.bfloat8_b, 1088) if mode == "b8_rne5" else (ttnn.bfloat4_b, 576)
    output = ttnn.allocate_tensor_on_device(src.shape, fmt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    tiles, tiles_per_head = src.volume() // 1024, src.shape[2] // 32 * 4
    size = device.compute_with_storage_grid_size()
    ncores = min(ncores, size.x * size.y, tiles // 4)
    coords = [ttnn.CoreCoord(i % size.x, i // size.x) for i in range(ncores)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    cbs = [
        ttnn.CBDescriptor(
            total_size=capacity * byte_count, core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=byte_count)],
        )
        for index, dtype, byte_count, capacity in (
            (0, ttnn.bfloat16, 2048, 8), (1, ttnn.bfloat16, 2048, 4), (16, fmt, out_bytes, 8)
        )
    ]
    reader, writer, compute = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, (offset, count) in zip(coords, core_segments(tiles, ncores)):
        reader[core.x][core.y] = [src.buffer_address(), bias.buffer_address(), offset, count]
        writer[core.x][core.y] = [output.buffer_address(), offset, count]
        compute[core.x][core.y] = [offset, count]
    desc = ttnn.ProgramDescriptor(cbs=cbs, semaphores=[], kernels=[
        ttnn.KernelDescriptor(
            kernel_source=PREFIX + "reader.cpp", core_ranges=grid,
            compile_time_args=[tiles_per_head] + ttnn.TensorAccessorArgs(src).get_compile_time_args()
            + ttnn.TensorAccessorArgs(bias).get_compile_time_args(),
            runtime_args=reader, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(
            kernel_source=PREFIX + "writer.cpp", core_ranges=grid,
            compile_time_args=ttnn.TensorAccessorArgs(output).get_compile_time_args(),
            runtime_args=writer, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(
            kernel_source=PREFIX + "compute.cpp", core_ranges=grid,
            compile_time_args=[int(mode == "b8_rne5"), tiles_per_head], runtime_args=compute,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, math_approx_mode=False)),
    ])
    return output, lambda: ttnn.generic_op([src, bias, output], desc), ncores


def l2_percent(actual, reference):
    reference = reference.double()
    norm = reference.norm()
    return float(100 * (actual.double() - reference).norm() / norm) if norm else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=("b8_rne5", "b4_rne"), default="b8_rne5")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--distribution", choices=("normal", "common_k", "wide", "zeros", "thresholds"), default="normal")
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trace-repeats", type=int, default=10)
    parser.add_argument("--host-only", action="store_true")
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0
    path = DIRECTORY / (args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    x, bias = make_input(args.length, args.heads, args.distribution, args.seed)
    centered = center_oracle(x, bias)
    expected = quantize_oracle(centered, args.mode)
    spilled = quantize_oracle(centered.bfloat16().float(), args.mode)
    spill_mismatches = int((expected != spilled).sum())
    print("CENTER_ORACLE_CHECK", args.mode, x.numel(), "BF16_SPILL_MISMATCHES", spill_mismatches, flush=True)
    if args.host_only:
        print("Host-only; no TTNN import or device opened.", flush=True)
        return
    import ttnn

    device = ttnn.open_device(device_id=0, trace_region_size=4194304 if args.iters else 0)
    try:
        src = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
        device_bias = ttnn.from_torch(bias, device=device, layout=ttnn.TILE_LAYOUT)
        output, invoke, cores = build(device, src, device_bias, args.cores, args.mode)
        invoke()
        actual = ttnn.to_torch(output).float()
        mismatches = int((actual != expected).sum())
        print("CENTER_DEVICE_CHECK", mismatches, flush=True)
        if mismatches:
            torch.save(dict(input=x, bias=bias, centered=centered, expected=expected, actual=actual),
                       DIRECTORY / (args.label + ".failure.pt"))
        assert not mismatches, f"Centered quantization mismatch count: {mismatches}"
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
            assert torch.equal(actual, ttnn.to_torch(output).float())
        tiles_per_head, tiles = args.length // 32 * 4, x.numel() // 1024
        segments = core_segments(tiles, cores)
        bias_loads = sum((start + count - 1) // tiles_per_head - start // tiles_per_head + 1
                         for start, count in segments)
        bytes_moved = tiles * (2048 + (1088 if args.mode == "b8_rne5" else 576)) + bias_loads * 8192
        median = statistics.median(times) if times else None
        files = [Path(__file__).resolve(), *sorted(DIRECTORY.glob("*.cpp")), *sorted(DIRECTORY.glob("*.hpp"))]
        record = dict(
            **vars(args), actual_cores=cores, actual_batch=4, numel=x.numel(),
            mismatches=mismatches, centered_quantization_l2_pct=l2_percent(actual, centered),
            bf16_center_spill_mismatches=spill_mismatches,
            bf16_center_spill_output_l2_pct=l2_percent(spilled, expected),
            core_segments=segments, bias_head_loads=bias_loads,
            median_ms=median, replay_ms=times,
            read_write_GBps=bytes_moved / (median * 1e6) if median else None,
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
