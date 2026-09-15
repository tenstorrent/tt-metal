# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated BF16-DST native/grid7 exp bit qualification, not attention.

Input is raw BF16 score delta in [-1000,0], scaled internally by1/sqrt(128).
Independent oracle: FP32-coefficient FMA, INT16 ties-away, shift17 (native15),
sign/pack ReLU, BF16 SFPU-store denormal flush and truncation. SFPMAD is
partially fused in hardware: any discrepancy at a threshold is a failure,
not an implicitly excluded sample. No NaN/Inf or BF16 input subnormals.
"""

import argparse
import hashlib
import json
import math
import statistics
import struct
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/exp_grid7_probe/"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


SCALE = f32(1 / math.sqrt(128))
SCALE_BITS = struct.unpack("<I", struct.pack("<f", SCALE))[0]


def coefficients(native=False):
    # Match each FP32 operation in the C++ constexpr expression.
    a = f32(f32(256.0 * f32(1.4426950408889634)) * SCALE)
    b = f32(32500.818359375)
    if not native:
        a, b = f32(a * 0.25), f32(b * 0.25)
    return a, b, 15 if native else 17


def oracle(src, native=False):
    assert src.dtype == torch.bfloat16
    x = src.float()
    assert bool(torch.isfinite(x).all()) and bool(((x >= -1000) & (x <= 0)).all())
    assert bool(((x == 0) | (x.abs() >= 2.0**-126)).all()), "Input subnormals excluded"
    a, b, shift = coefficients(native)
    # Exact BF16*FP32 product and addition fit FP64 over this bounded range;
    # one FP32 rounding models the intended SFPU FMA result.
    transformed = (x.double() * a + b).float()
    magnitude = (transformed.double().abs() + 0.5).floor().clamp_max(32767).long()
    bits = (magnitude << shift) & 0x7FFFFFFF
    # Negative results are zeroed by pack ReLU. SFPSTORE BF16 flushes
    # exponent-zero results (including grid-generated subnormals) first.
    bits = torch.where((transformed < 0) | ((bits & 0x7F800000) == 0), 0, bits)
    expected = (bits >> 16).to(torch.int16).contiguous().view(torch.bfloat16)
    assert bool(torch.isfinite(expected).all()) and bool((expected >= 0).all())
    return expected, transformed, magnitude


def input_values(shape, distribution, seed, native=False):
    count = math.prod(shape)
    generator = torch.Generator().manual_seed(seed)
    if distribution == "normal":
        return (-torch.randn(shape, generator=generator).abs() * 64).clamp_min(-1000).bfloat16()
    raw = torch.arange(65536, dtype=torch.int32)
    all_values = raw.to(torch.int16).view(torch.bfloat16)
    xf = all_values.float()
    selected = torch.isfinite(xf) & (xf >= -1000) & (xf <= 0)
    selected &= (xf == 0) | (xf.abs() >= 2.0**-126)
    finite = all_values[selected]
    a, b, _ = coefficients(native)
    targets = torch.arange(math.ceil(b) + 1, dtype=torch.float64) + 0.5
    threshold = ((targets - b) / a).bfloat16()
    # Neighboring BF16 representations around each integer-rounding boundary,
    # not FP64 boundaries silently fed to a BF16 kernel.
    bits = threshold.view(torch.int16).int()
    neighbors = torch.cat([(bits + offset).to(torch.int16).view(torch.bfloat16) for offset in (-1, 0, 1)])
    nf = neighbors.float()
    neighbors = neighbors[torch.isfinite(nf) & (nf >= -1000) & (nf <= 0)]
    neighbors = neighbors[(neighbors == 0) | (neighbors.float().abs() >= 2.0**-126)]
    zeros = torch.tensor([0, -32768], dtype=torch.int16).view(torch.bfloat16)
    underflow = finite[finite.float() < -960]
    if distribution == "finite_bf16":
        values = finite
    elif distribution == "thresholds":
        values = neighbors
    elif distribution == "underflow":
        values = underflow
    elif distribution == "zeros":
        values = zeros
    else:
        assert distribution == "all"
        random_values = (-torch.randn(32768, generator=generator).abs() * 64).clamp_min(-1000).bfloat16()
        values = torch.cat([finite, neighbors, underflow, zeros, random_values])
    assert count >= values.numel(), f"Need at least{values.numel()} elements for complete selected pool"
    values = values.repeat((count + values.numel() - 1) // values.numel())[:count]
    return values.reshape(shape).contiguous()


def check(actual, src, native=False):
    expected, transformed, magnitude = oracle(src, native)
    actual = actual.bfloat16().contiguous()
    ab, eb = actual.view(torch.int16), expected.view(torch.int16)
    mismatch_mask = ab != eb
    ix = mismatch_mask.flatten().nonzero().flatten()[:16]
    tuples = []
    for i in ix.tolist():
        tuples.append(
            dict(
                index=i,
                input=float(src.flatten()[i]),
                input_bits=f"0x{int(src.view(torch.int16).flatten()[i]) & 0xffff:04x}",
                transformed=float(transformed.flatten()[i]),
                transformed_bits=f"0x{int(transformed.view(torch.int32).flatten()[i]) & 0xffffffff:08x}",
                rounded_magnitude=int(magnitude.flatten()[i]),
                expected_bits=f"0x{int(eb.flatten()[i]) & 0xffff:04x}",
                actual_bits=f"0x{int(ab.flatten()[i]) & 0xffff:04x}",
            )
        )
    positive = actual > 0
    ties = transformed.double().abs().frac() == 0.5
    record = dict(
        numel=src.numel(),
        unique_input_bits=int(torch.unique(src.view(torch.int16)).numel()),
        mismatch=int(mismatch_mask.sum()),
        mismatch_examples=tuples,
        finite=bool(torch.isfinite(actual).all()),
        nonnegative=bool((actual >= 0).all()),
        positive_count=int(positive.sum()),
        zero_count=int((actual == 0).sum()),
        positive_low_bf16_mantissa_bit_nonzero=int((positive & ((ab.int() & 1) != 0)).sum()),
        transformed_exact_half_ties=int(ties.sum()),
        negative_transformed=int((transformed < 0).sum()),
        pre_store_subnormal_count=int(
            ((magnitude > 0) & (magnitude < (256 if native else 64)) & (transformed >= 0)).sum()
        ),
        coefficients=dict(
            a=coefficients(native)[0],
            b=coefficients(native)[1],
            shift=coefficients(native)[2],
            scale=SCALE,
            scale_bits=f"0x{SCALE_BITS:08x}",
        ),
        output_sha256=hashlib.sha256(ab.numpy().tobytes()).hexdigest(),
        input_sha256=hashlib.sha256(src.view(torch.int16).numpy().tobytes()).hexdigest(),
    )
    print("GRID7_PROBE_CHECK", json.dumps(record), flush=True)
    assert record["finite"] and record["nonnegative"]
    assert record["mismatch"] == 0, "Exact exp-grid bit model mismatch; see pre-assert diagnostic"
    if not native:
        assert record["positive_low_bf16_mantissa_bit_nonzero"] == 0, "Grid7 positive P is not E8M6"
    return record


def source_files():
    return sorted(
        set(
            [
                Path(__file__).resolve(),
                HERE / "exp_grid7.hpp",
                *sorted((HERE / "exp_grid7_probe").glob("*.cpp")),
                HERE / "preprocess/reader.cpp",
                HERE / "preprocess/writer.cpp",
                ROOT
                / "experiments/sdpa-l2/single-core-resident-v1/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp",
                ROOT
                / "experiments/sdpa-l2/single-core-resident-v1/main/tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h",
                ROOT / "tt_metal/hw/inc/api/compute/eltwise_unary/exp.h",
                ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h",
                ROOT / "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h",
            ]
        )
    )


def build(device, src, ncores=1, batch=4, native=False):
    """Return (preallocated BF16 output, recompute callable, actual core count)."""
    assert src.dtype == ttnn.bfloat16 and ncores > 0
    assert len(src.shape) == 4 and src.shape[-1] == 128 and src.shape[-2] % 32 == 0
    assert batch in (1, 2, 4)
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
                compile_time_args=[SCALE_BITS, batch],
                runtime_args=compute,
                defines=[("EXP_APPROX_MODE", "1")] + ([] if native else [("SDPA_LOFI_EXP_GRID7", "1")]),
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, math_approx_mode=True
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
    parser.add_argument("--native", action="store_true", help="Unchanged native256-step grid instead of64")
    parser.add_argument(
        "--distribution", choices=("all", "normal", "thresholds", "underflow", "zeros", "finite_bf16"), default="all"
    )
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--batch", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trace-repeats", type=int, default=10)
    args = parser.parse_args()
    assert args.length > 0 and args.length % 32 == 0
    assert args.cores > 0 and args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0
    assert Path(args.label).name == args.label
    path = HERE / (args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(4)
    src = input_values((1, 1, args.length, 128), args.distribution, args.seed, args.native)
    expected, _, _ = oracle(src, args.native)
    files = source_files()
    pins = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    device = ttnn.open_device(device_id=0, trace_region_size=4194304 if args.iters else 0)
    try:
        tensor = ttnn.from_torch(src, device=device, layout=ttnn.TILE_LAYOUT)
        out, invoke, cores = build(device, tensor, args.cores, args.batch, args.native)
        invoke()
        actual = ttnn.to_torch(out)
        result = check(actual, src, args.native)
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
                        times.append(1000 * (time.perf_counter() - start) / args.trace_repeats)
            finally:
                ttnn.release_trace(device, trace)
            assert torch.equal(actual.view(torch.int16), ttnn.to_torch(out).view(torch.int16))
        assert pins == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        record = dict(
            **vars(args),
            **result,
            actual_cores=cores,
            source_sha256=pins,
            expected_output_sha256=hashlib.sha256(expected.view(torch.int16).numpy().tobytes()).hexdigest(),
            median_ms=statistics.median(times) if times else None,
            replay_ms=times,
            trace_equal=True if times else None,
            source_unchanged=True,
            contract=__doc__,
            destination="BF16",
            output_format="BF16",
            attention_test=False,
            denominator_or_pv_test=False,
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("GRID7_PROBE_RESULT", json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
