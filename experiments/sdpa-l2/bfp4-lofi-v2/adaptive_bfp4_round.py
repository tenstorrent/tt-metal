# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated native BFP4 exponent search with an exact FP32 arithmetic oracle.

BF16 finite normal/zero input, group16, maximum exponents [-123,106].
FP32 DST only: tile0 original/result, tile1 best value, tile2 score/scale.
No attention kernels, frozen quantizers, or existing source are modified.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DIRECTORY = HERE / "adaptive_bfp4_round"
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/"
SPEC = importlib.util.spec_from_file_location("adaptive_qualified_b4", HERE / "bfp4_round.py")
QUALIFIED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFIED)
SEARCH = {"baseline": 0, "minus": 1, "pm": 2}
DISTRIBUTIONS = ("normal", "thresholds", "ties", "wide", "zeros", "group_outliers")


def validate_input(x):
    QUALIFIED.validate_input(x)
    maximum = x.float().abs().reshape(-1, 16).amax(-1)
    assert bool(((maximum == 0) | (maximum >= 2.0**-123)).all()), (
        "E-1 candidate needs E>=-123 so every final B4 quantum is FP32 normal"
    )


def ftz(x):
    return torch.where(x.abs() < 2.0**-126, 0.0, x)


def score_tree(error):
    """Separate FP32 squares; pair even/odd; cyclic ror1/2/4; retain lane0.

    Every addition rounds independently. Other seven lane sums are discarded,
    and lane0 is broadcast before selection, as in the raw-instruction kernel.
    """
    square = ftz(error * error)
    lanes = ftz(square[:, 0::2] + square[:, 1::2])
    for rotation in (1, 2, 4):
        lanes = ftz(lanes + torch.roll(lanes, rotation, -1))
    return lanes[:, :1]


def oracle(x, search="pm"):
    validate_input(x)
    groups = x.float().reshape(-1, 16)
    maximum = groups.abs().amax(-1, keepdim=True)
    exponent = torch.frexp(maximum)[1] - 1
    exponent = torch.where(maximum == 0, 0, exponent)
    scale = torch.ldexp(torch.ones_like(maximum), exponent)
    inverse = torch.ldexp(torch.ones_like(maximum), -exponent)
    normalized = ftz(groups.abs() * inverse)
    offsets = (0,) if search == "baseline" else (0, -1) if search == "minus" else (0, -1, 1)
    selected = torch.zeros_like(exponent)
    fp64_selected = torch.zeros_like(exponent)
    candidates, scores, scores64 = {}, {}, {}
    for offset in offsets:
        magic = 2.0 ** (21 + offset)
        cap = 1.75 * 2.0**offset
        candidate = ((normalized + magic) - magic).clamp_max(cap)
        error = ftz(normalized - candidate)
        score = score_tree(error)
        # Independent true reconstructed-MSE control, not the acceptance oracle.
        candidate64 = (groups.double().abs() / scale.double() / (2.0 ** (offset - 2))).round().clamp_max(7)
        candidate64 *= 2.0 ** (offset - 2)
        assert torch.equal(candidate.double(), candidate64)
        score64 = (groups.double().abs() / scale.double() - candidate64).square().sum(-1, keepdim=True)
        candidates[offset], scores[offset], scores64[offset] = candidate, score, score64
        if offset == 0:
            best, best64 = score, score64
            result = candidate.clone()
        else:
            improve = score < best
            result = torch.where(improve, candidate, result)
            selected = torch.where(improve, offset, selected)
            best = torch.minimum(best, score)
            improve64 = score64 < best64
            fp64_selected = torch.where(improve64, offset, fp64_selected)
            best64 = torch.minimum(best64, score64)
    output = (result * scale * groups.sign()).reshape_as(x)
    if search == "baseline":
        assert torch.equal(output, QUALIFIED.host_rne_bfp4(x))
    # E+1 may intentionally emit exponent107, beyond the older preprocessor's
    # magic-constant input contract. Check the native B4 grid directly rather
    # than weakening that frozen validation or reusing its restricted oracle.
    out_groups = output.reshape(-1, 16)
    out_shared = torch.frexp(out_groups.abs().amax(-1, keepdim=True))[1] - 1
    out_step = torch.ldexp(torch.ones_like(maximum), out_shared - 2)
    repacked = (out_groups.abs() / out_step + 0.5).floor().clamp_max(7) * out_step * out_groups.sign()
    assert torch.equal(output.bfloat16().float(), output)
    assert torch.equal(output, repacked.reshape_as(output))
    active = maximum != 0
    out_exp = torch.frexp(output.reshape(-1, 16).abs().amax(-1, keepdim=True))[1] - 1
    assert bool(((out_exp == exponent + selected) | ~active).all())
    stats = dict(
        groups=groups.shape[0],
        selected_counts={str(o): int((selected == o).sum()) for o in (-1, 0, 1)},
        selection_differs_fp64_groups=int((selected != fp64_selected).sum()),
        baseline_tie_counts={str(o): int((scores[o] == scores[0]).sum()) for o in offsets if o != 0},
        induced_exponent_mismatches=0,
        native_grid_roundtrip_mismatches=0,
        score_contract="FP32 power-of-two normalized magnitudes, separate FTZ subtract/square/add, even+odd then cyclic ror1/2/4; only lane0 retained/broadcast; strict less-than; candidate order0,-1,+1",
    )
    return output, stats


def make_input(length, distribution, seed):
    assert length > 0 and length % 32 == 0
    if distribution in ("normal", "thresholds", "zeros"):
        return QUALIFIED.make_input(length, distribution, seed)
    shape = (1, 1, length, 128)
    count = length * 128 // 16
    group = torch.arange(count)
    lane = torch.arange(16)
    if distribution == "group_outliers":
        gen = torch.Generator().manual_seed(seed)
        values = torch.randn((count, 16), generator=gen)
        values[group, group % 16] *= 32
        values[::97] = 0
        return values.reshape(shape).bfloat16()
    if distribution == "ties":
        values = ((group[:, None] + lane[None, :] * 17) % 256).float() / 128
        values[group, group % 16] = 1.75
        # Exactly represented by baseline and E+1: tie must keep baseline.
        values[::3] = 1.0
        values[::97] = 0
        exponent = (group % 17 - 8).int()
    elif distribution == "wide":
        mantissa = ((group[:, None] + lane[None, :] * 19) % 128).float()
        delta = (lane % 7).int()
        values = torch.ldexp(1 + mantissa / 128, -delta[None, :])
        exponent = (group % 230 - 123).int()
        values = torch.where(exponent[:, None] - delta[None, :] < -126, 0.0, values)
        values[group, group % 16] = 1.75
        # Also exercise exponent changes at the supported range boundaries,
        # not just baseline-anchored groups whose maximum is always1.75.
        bulk = ((group[:, None] + lane[None, :] * 17) % 128).float() / 128
        values[1::3] = bulk[1::3]
        values[group[1::3], (group % 16)[1::3]] = 1 + (group[1::3] % 8).float() / 128
        values[2::3] = bulk[2::3] / 32
        values[group[2::3], (group % 16)[2::3]] = 1.9375
        values[::97] = 0
    else:
        raise ValueError(distribution)
    values = torch.ldexp(values, exponent[:, None])
    values = torch.where(values.abs() < 2.0**-126, 0.0, values)
    sign = torch.where(((group[:, None] + lane[None, :]) & 1) != 0, -1.0, 1.0)
    return (values * sign).reshape(shape).bfloat16()


def source_pins():
    paths = [Path(__file__), DIRECTORY / "compute.cpp", HERE / "bfp4_round.py",
             HERE / "bfp4_round/reader.cpp", HERE / "bfp4_round/writer.cpp"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def build(device, src, ncores=1, search="pm", output_format="b4"):
    import ttnn

    assert src.dtype == ttnn.bfloat16
    assert src.shape[-1] % 32 == 0 and src.shape[-2] % 32 == 0
    assert ncores > 0 and search in SEARCH and output_format in ("b4", "bf16")
    dtype, size = (ttnn.bfloat4_b, 576) if output_format == "b4" else (ttnn.bfloat16, 2048)
    out = ttnn.allocate_tensor_on_device(src.shape, dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    tiles = src.volume() // 1024
    grid_size = device.compute_with_storage_grid_size()
    ncores = min(ncores, tiles, grid_size.x * grid_size.y)
    coords = [ttnn.CoreCoord(i % grid_size.x, i // grid_size.x) for i in range(ncores)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * page,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
        )
        for index, fmt, page in ((0, ttnn.bfloat16, 2048), (16, dtype, size))
    ]
    reader, writer, compute = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    offset = 0
    for i, core in enumerate(coords):
        count = tiles // ncores + (i < tiles % ncores)
        reader[core.x][core.y] = [src.buffer_address(), offset, count]
        writer[core.x][core.y] = [out.buffer_address(), offset, count]
        compute[core.x][core.y] = [count]
        offset += count
    desc = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=[],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "bfp4_round/reader.cpp",
                core_ranges=grid,
                compile_time_args=[1] + ttnn.TensorAccessorArgs(src).get_compile_time_args(),
                runtime_args=reader,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "bfp4_round/writer.cpp",
                core_ranges=grid,
                compile_time_args=[1] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=writer,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "adaptive_bfp4_round/compute.cpp",
                core_ranges=grid,
                compile_time_args=[SEARCH[search]],
                runtime_args=compute,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False,
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
    parser.add_argument("--search", choices=SEARCH, default="pm")
    parser.add_argument("--output-format", choices=("b4", "bf16"), default="b4")
    parser.add_argument("--distribution", choices=DISTRIBUTIONS, default="normal")
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
    torch.set_num_threads(4)
    pins = source_pins()
    x = make_input(args.length, args.distribution, args.seed)
    expected, oracle_stats = oracle(x, args.search)
    print("ADAPTIVE_BFP4_ORACLE", json.dumps(oracle_stats), flush=True)
    if args.host_only:
        print("Host-only arithmetic/encoding assertions passed; no device opened.", flush=True)
        return

    import ttnn

    device = ttnn.open_device(device_id=0, trace_region_size=4194304 if args.iters else 0)
    try:
        src = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
        out, invoke, cores = build(device, src, args.cores, args.search, args.output_format)
        invoke()
        actual = ttnn.to_torch(out).float()
        mismatch = int((actual != expected).sum())
        # Native BFP zero does not retain an IEEE signed-zero payload. Require
        # exact decoded FP32 bits after explicitly canonicalizing zeros only.
        actual_bits = torch.where(actual == 0, 0.0, actual).contiguous().view(torch.int32)
        expected_bits = torch.where(expected == 0, 0.0, expected).contiguous().view(torch.int32)
        bit_mismatch = int((actual_bits != expected_bits).sum())
        print("ADAPTIVE_BFP4_CHECK", mismatch, actual.numel(), flush=True)
        if mismatch:
            torch.save(dict(input=x, actual=actual, expected=expected, oracle_stats=oracle_stats),
                       DIRECTORY / (args.label + ".failure.pt"))
        assert mismatch == 0 and bit_mismatch == 0, f"{mismatch} adaptive BFP4 arithmetic-oracle mismatches"
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
        output_bytes = x.numel() // 1024 * (576 if args.output_format == "b4" else 2048)
        byte_count = x.numel() * 2 + output_bytes
        assert source_pins() == pins, "Sources changed during measurement"
        record = dict(
            **vars(args), actual_cores=cores, mismatch=mismatch, decoded_bit_mismatch=bit_mismatch, numel=x.numel(),
            oracle=oracle_stats, source_sha256=pins,
            fp32_dst=True, dst_tiles_reserved=3, batch=1, cb_payload_per_core=4096 + 2 * (576 if args.output_format == "b4" else 2048),
            median_ms=median, replay_ms=times, read_write_GBps=byte_count / (median * 1e6) if median else None,
            actual_sha256=hashlib.sha256(actual_bits.numpy().tobytes()).hexdigest(),
            expected_sha256=hashlib.sha256(expected_bits.numpy().tobytes()).hexdigest(),
            hash_contract="Decoded FP32 bits with signed zeros canonicalized, not raw packed DRAM bytes",
            quantization_l2_pct=float(100 * (actual.double() - x.double()).norm() / x.double().norm()) if x.double().norm() else 0.0,
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
