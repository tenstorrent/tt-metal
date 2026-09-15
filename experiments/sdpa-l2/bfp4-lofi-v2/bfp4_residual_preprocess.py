# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused 2/3-component native-group RNE BFP4 decomposition of BF16 inputs.

Reads input once; residual subtraction remains in FP32 SFPU registers. Outputs
use BFP4, optionally with an RNE5/BFP8 second component. CPU oracles check each
intermediate residual must satisfy the finite ordinary-range contract.
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
DIRECTORY = HERE / "bfp4_residual_preprocess"
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/bfp4_residual_preprocess/"


def validate_input(x):
    assert x.dtype == torch.bfloat16, "This prototype's exact oracle contract is BF16 input"
    assert x.shape[-1] % 16 == 0
    values = x.float()
    assert bool(torch.isfinite(values).all()), "NaN/Inf are outside the prototype contract"
    magnitude = values.abs()
    assert bool(((magnitude == 0) | (magnitude >= 2.0**-126)).all()), "No BF16 subnormal inputs"
    maximum = magnitude.reshape(-1, 16).amax(-1)
    assert bool(
        ((maximum == 0) | ((maximum >= 2.0**-124) & (maximum < 2.0**107))).all()
    ), "Nonzero native-group maximum exponents must be in [-124, 106]"


def host_rne_bfp4(x):
    """Direct shared-exponent 3-magnitude-bit RNE + saturation, host semantics.

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
    integer = aligned >> 21
    remainder = aligned & ((1 << 21) - 1)
    halfway = 1 << 20
    increment = (remainder > halfway) | ((remainder == halfway) & ((integer & 1) != 0))
    integer = (integer + increment.long()).clamp_max(7)
    quantized = torch.ldexp(integer.float(), (shared - 129).int())
    return (quantized * groups.sign()).reshape_as(x)


def magic_rne_bfp4(x):
    """FP32 arithmetic oracle for the actual kernel, including all-zero groups."""
    validate_input(x)
    groups = x.float().reshape(-1, 16)
    magnitude = groups.abs()
    maximum = magnitude.amax(-1, keepdim=True)
    exponent = torch.frexp(maximum)[1] - 1
    # The kernel uses biased exponent zero for all-zero groups. Both choices
    # below yield zero; choose ordinary constants to avoid host FTZ dependence.
    exponent = torch.where(maximum == 0, 0, exponent)
    magic = torch.ldexp(torch.ones_like(maximum), exponent + 21)
    cap = torch.ldexp(torch.full_like(maximum, 1.75), exponent)
    rounded = (magnitude + magic) - magic
    return (torch.minimum(rounded, cap) * groups.sign()).reshape_as(x)


def rne5(x):
    """Per-value round-to-nearest-even to five total significant bits."""
    raw = x.float().contiguous().view(torch.int32).long() & 0xFFFFFFFF
    rounded = (raw + 0x3FFFF + ((raw >> 19) & 1)) & 0xFFF80000
    return rounded.int().view(torch.float32)


def native_bfp8_rne5(x):
    """Independent integer model of RNE5 followed by native BFP8 packing.

    RNE5 is exactly representable in the packer's E8M6 intermediate. Native
    BFP8 then aligns to the group exponent and rounds nearest, ties AWAY,
    saturating at magnitude 127. This is not the host BFP8 ties-even oracle.
    No artificial exponent clamp or floating-point scale is needed.
    """
    validate_input(x)
    groups = rne5(x).reshape(-1, 16)
    raw = groups.abs().contiguous().view(torch.int32).long()
    exponent = (raw >> 23) & 255
    shared = exponent.amax(-1, keepdim=True)
    # BF16's eight-bit significand is exact here (RNE5 has three zero bits).
    significand = torch.where(exponent == 0, 0, ((raw >> 16) & 127) | 128)
    shift = shared - exponent + 1
    safe_shift = shift.clamp_max(9)
    integer = (significand + (1 << (safe_shift - 1))) >> safe_shift
    integer = torch.where(shift > 8, 0, integer).clamp_max(127)
    return (torch.ldexp(integer.float(), (shared - 133).int()) * groups.sign()).reshape_as(x)


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
        values[group, group % 16] = 1.75
        exponent = (group % 17 - 8).int()
    elif distribution == "wide":
        # All 128 BF16 mantissas, both signs, and native-group exponents
        # throughout [-90, 90] (leaving recursive-residual headroom), mixed with much smaller same-group
        # values. This deliberately does not use the v1 -100 exponent clamp.
        mantissa = ((group[:, None] + lane[None, :] * 19) % 128).float()
        values = 1 + mantissa / 128
        delta = (lane % 7).int()
        values = torch.ldexp(values, -delta[None, :])
        values[group, group % 16] = 1.75
        exponent = (group % 181 - 90).int()
        # Avoid individual BF16 subnormals in low-exponent groups.
        values = torch.where((exponent[:, None] - delta[None, :] < -126), torch.zeros_like(values), values)
        values[group, group % 16] = 1.75
    else:
        raise ValueError(distribution)
    values = torch.ldexp(values, exponent[:, None])
    sign = torch.where(((group[:, None] + lane[None, :]) & 1) != 0, -1.0, 1.0)
    values *= sign
    # Exact zero groups test max=0 and avoid treating the sign bit as magnitude.
    values[::97] = 0
    return values.reshape(shape).bfloat16()


def component_oracle(x, components, second_format="b4"):
    assert components in (2, 3)
    assert second_format in ("b4", "b8_rne5")
    assert second_format == "b4" or components == 2
    residual = x.float()
    result = []
    for stage in range(components):
        # This preprocessor intentionally excludes underflowing intermediate
        # groups rather than silently depending on SFPU flush-to-zero behavior.
        narrow = residual.bfloat16()
        assert torch.equal(narrow.float(), residual), f"Stage {stage}: residual is not exactly BF16"
        validate_input(narrow)
        if stage == 1 and second_format == "b8_rne5":
            quantized = native_bfp8_rne5(narrow)
        else:
            quantized = host_rne_bfp4(narrow)
            assert torch.equal(quantized, magic_rne_bfp4(narrow)), f"Stage {stage}: CPU oracle disagreement"
        result.append(quantized)
        residual = residual - quantized
    return result


def build(device, src, components=2, ncores=1, batch=None, fp32_dst=False, second_format="b4"):
    """Return (outputs, invoke, actual_cores); optional second_format='b8_rne5'."""
    import ttnn

    assert components in (2, 3) and src.dtype == ttnn.bfloat16 and ncores > 0
    assert second_format in ("b4", "b8_rne5")
    assert second_format == "b4" or components == 2
    second_b8 = second_format == "b8_rne5"
    capacity = 4 if fp32_dst else 8
    batch = capacity // components if batch is None else batch
    assert 1 <= batch and components * batch <= capacity
    assert src.shape[-1] % 32 == 0 and src.shape[-2] % 32 == 0
    tiles = src.volume() // 1024
    assert tiles % batch == 0
    output_formats = [
        (ttnn.bfloat8_b, 1088) if c == 1 and second_b8 else (ttnn.bfloat4_b, 576) for c in range(components)
    ]
    outputs = [
        ttnn.allocate_tensor_on_device(src.shape, fmt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
        for fmt, _ in output_formats
    ]
    size = device.compute_with_storage_grid_size()
    ncores = min(ncores, tiles // batch, size.x * size.y)
    coords = [ttnn.CoreCoord(i % size.x, i // size.x) for i in range(ncores)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    formats = [(0, ttnn.bfloat16, 2048)] + [
        (16 + c, fmt, byte_count) for c, (fmt, byte_count) in enumerate(output_formats)
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * batch * byte_count,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=byte_count)],
        )
        for index, fmt, byte_count in formats
    ]
    reader, writer, compute = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addresses = [t.buffer_address() for t in outputs] + [0] * (3 - components)
    offset = 0
    for i, core in enumerate(coords):
        count = (tiles // batch // ncores + (i < tiles // batch % ncores)) * batch
        reader[core.x][core.y] = [src.buffer_address(), offset, count]
        writer[core.x][core.y] = addresses + [offset, count]
        compute[core.x][core.y] = [count]
        offset += count
    # Each accessor embeds its own aligned page size (BFP4=576, BFP8=1088).
    # Supply a harmless third descriptor in the two-component specialization.
    writer_compile_args = [components, batch]
    for tensor in outputs + ([outputs[0]] if components == 2 else []):
        writer_compile_args += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
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
                compile_time_args=writer_compile_args,
                runtime_args=writer,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "compute.cpp",
                core_ranges=grid,
                compile_time_args=[components, batch, int(second_b8)],
                runtime_args=compute,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=fp32_dst, math_approx_mode=False
                ),
            ),
        ],
    )
    return outputs, lambda: ttnn.generic_op([src, *outputs], desc), ncores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--components", type=int, choices=(2, 3), default=2)
    parser.add_argument("--second-format", choices=("b4", "b8_rne5"), default="b4")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--fp32-dst", action="store_true")
    parser.add_argument("--distribution", choices=("normal", "thresholds", "wide", "zeros"), default="normal")
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
    x = make_input(args.length, args.distribution, args.seed)
    expected = component_oracle(x, args.components, args.second_format)
    print("RESIDUAL_ORACLE_CHECK", args.components, args.second_format, x.numel(), flush=True)
    if args.host_only:
        print("Host-only CPU oracle agreement; no device opened.", flush=True)
        return
    import ttnn

    device = ttnn.open_device(device_id=0, trace_region_size=4194304 if args.iters else 0)
    try:
        src = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
        outputs, invoke, cores = build(
            device, src, args.components, args.cores, args.batch, args.fp32_dst, args.second_format
        )
        invoke()
        actual = [ttnn.to_torch(t).float() for t in outputs]
        mismatches = [int((a != e).sum()) for a, e in zip(actual, expected)]
        print("RESIDUAL_COMPONENT_CHECK", mismatches, flush=True)
        if any(mismatches):
            torch.save(dict(input=x, actual=actual, expected=expected), DIRECTORY / (args.label + ".failure.pt"))
        assert not any(mismatches), f"Component mismatches: {mismatches}"
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
            assert all(torch.equal(a, ttnn.to_torch(o).float()) for a, o in zip(actual, outputs))
        median = statistics.median(times) if times else None
        output_bytes_per_tile = 576 * args.components + (512 if args.second_format == "b8_rne5" else 0)
        byte_count = x.numel() * 2 + x.numel() // 1024 * output_bytes_per_tile
        reference = x.double()
        norm = reference.norm()
        prefix_errors, reconstruction = [], torch.zeros_like(reference)
        for value in actual:
            reconstruction += value.double()
            prefix_errors.append(float(100 * (reconstruction - reference).norm() / norm) if norm else 0.0)
        source_files = [Path(__file__).resolve(), *sorted(DIRECTORY.glob("*.cpp")), *sorted(DIRECTORY.glob("*.hpp"))]
        record = dict(
            **vars(args),
            actual_cores=cores,
            actual_batch=args.batch or ((4 if args.fp32_dst else 8) // args.components),
            component_mismatches=mismatches,
            numel=x.numel(),
            prefix_reconstruction_l2_pct=prefix_errors,
            median_ms=median,
            replay_ms=times,
            read_write_GBps=byte_count / (median * 1e6) if median else None,
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
