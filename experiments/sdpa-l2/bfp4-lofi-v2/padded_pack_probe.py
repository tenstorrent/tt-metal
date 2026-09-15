# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""BF16 width4 padded-page pack MOP qualification, not attention benchmarking.
128 distinct source tiles; compact scalar/width4 and padded scalar/custom4.
Custom4 uses output Y stride and per-tile Last with one C++ pack call per4.
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
DIRECTORY = HERE / "padded_pack_probe"
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/padded_pack_probe/"
SOURCE_PATHS = [Path(__file__).resolve(), *sorted(DIRECTORY.glob("*.cpp")),
    ROOT / "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h",
    ROOT / "tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h",
    ROOT / "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_template.h",
    ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_common_api.h"]


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE_PATHS}


def make_input(seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((1, 1, 128 * 32, 32), generator=generator).bfloat16()
    for tile in range(128):
        x[0, 0, tile * 32, 0] = tile + 1
        x[0, 0, tile * 32 + 16, 16] = -(tile + 1)
    return x


def build(device, src, args, page_bytes, mode):
    batch = 4 if args.fp32_dst else 8
    dtype = ttnn.bfloat16
    assert (mode in (0, 1) and page_bytes == 2048) or (mode in (0, 2) and page_bytes == 4096)
    out = ttnn.allocate_tensor_on_device(
        src.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cbs = [
        ttnn.CBDescriptor(
            total_size=tiles * size, core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=size)],
        )
        for index, tiles, fmt, size in (
            (0, 128, ttnn.bfloat16, 2048),
            (8, 2 * batch, dtype, page_bytes),
            (16, 2 * batch, ttnn.bfloat16, 2048),
        )
    ]
    reader, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader[0][0] = [src.buffer_address()]
    writer[0][0] = [out.buffer_address()]
    cta = [batch, args.tiles, mode]
    descriptor = ttnn.ProgramDescriptor(
        cbs=cbs, semaphores=[],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "reader.cpp", core_ranges=grid,
                compile_time_args=cta + ttnn.TensorAccessorArgs(src).get_compile_time_args(),
                runtime_args=reader, config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "writer.cpp", core_ranges=grid,
                compile_time_args=cta + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=writer, config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "compute.cpp", core_ranges=grid, compile_time_args=cta,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=args.fp32_dst,
                    dst_full_sync_en=False, math_approx_mode=False)),
        ],
    )
    return out, lambda: ttnn.generic_op([src, out], descriptor), 128 * 2048 + 2 * batch * (page_bytes + 2048)


def timed(device, invoke, args):
    if not args.iters:
        return dict(median_ms=None, replay_ms_per_invocation=[])
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(args.trace_repeats):
        invoke()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    times = []
    try:
        for iteration in range(args.warmup + args.iters):
            start = time.perf_counter()
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            if iteration >= args.warmup:
                times.append(1000 * (time.perf_counter() - start) / args.trace_repeats)
    finally:
        ttnn.release_trace(device, trace)
    return dict(median_ms=statistics.median(times), replay_ms_per_invocation=times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--fp32-dst", action="store_true")
    parser.add_argument("--tiles", type=int, default=128, help="Multiple of 128; use 8192 or more for timing")
    parser.add_argument("--seed", type=int, default=1250)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-repeats", type=int, default=10)
    parser.add_argument("--clock-mhz", type=float, default=1200, help="Assumed clock; not queried/measured")
    args = parser.parse_args()
    assert Path(args.label).name == args.label, "Use a plain fresh label"
    assert args.tiles >= 128 and args.tiles % 128 == 0
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0 and args.clock_mhz > 0
    path, matrices_path = DIRECTORY / (args.label + ".json"), DIRECTORY / (args.label + ".pt")
    assert not path.exists() and not matrices_path.exists(), "Use a fresh label"
    torch.set_num_threads(4)
    hashes_before = source_hashes()
    x = make_input(args.seed)
    native_bytes = 2048
    device = ttnn.open_device(device_id=0, trace_region_size=4194304 if args.iters else 0)
    try:
        src = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
        cases, matrices = [], {"input_bf16": x}
        for name, page_bytes, mode in (("compact_scalar", 2048, 0), ("compact_width4", 2048, 1),
                                      ("padded_scalar", 4096, 0), ("padded_custom4", 4096, 2)):
            out, invoke, cb_bytes = build(device, src, args, page_bytes, mode)
            invoke()
            actual = ttnn.to_torch(out)
            matrices[name + "_output_bf16"] = actual
            finite = bool(torch.isfinite(actual).all())
            mismatch = int((actual != x).sum())
            error = actual.double() - x.double()
            case = dict(
                name=name, page_bytes=page_bytes, native_payload_bytes=native_bytes,
                pack_width=1 if mode == 0 else 4, custom_mop=mode == 2,
                cb_bytes_per_core=cb_bytes, finite=finite, mismatch_vs_input=mismatch,
                max_absolute_error=float(error.abs().max()) if finite else None,
                quantization_l2_pct=float(100 * error.norm() / x.double().norm()) if finite else None,
                output_sha256=hashlib.sha256(actual.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest(),
            )
            cases.append((case, out, invoke))
            print("ROUNDTRIP_CHECK", json.dumps(case), flush=True)
        compact = matrices["compact_scalar_output_bf16"]
        control_mismatch = {c["name"]: int((matrices[c["name"] + "_output_bf16"] != compact).sum())
                            for c, _, _ in cases}
        exact_required = True
        correctness_pass = all(c["finite"] and c["mismatch_vs_input"] == 0 for c, _, _ in cases)
        # Preserve complete matrices before raising any correctness failure.
        torch.save(matrices, matrices_path)
        if correctness_pass:
            for case, out, invoke in cases:
                case.update(timed(device, invoke, args))
                ms = case["median_ms"]
                rate = args.tiles * 1000 / ms if ms else None
                case.update(
                    roundtrip_tiles_per_second=rate,
                    nominal_l1_payload_GBps=rate * (4096 + 2 * native_bytes) / 1e9 if rate else None,
                    estimated_cycles_per_roundtrip_tile=args.clock_mhz * 1e6 / rate if rate else None,
                    trace_equal=True if args.iters else None,
                )
                assert torch.equal(matrices[case["name"] + "_output_bf16"], ttnn.to_torch(out))
        record = dict(
            **vars(args), cores=1, batch=4 if args.fp32_dst else 8,
            input_shape=list(x.shape), output_shape=list(compact.shape),
            resident_input_tiles=128, saved_output_tiles=128,
            exact_required=exact_required, correctness_pass=bool(correctness_pass),
            padded_vs_compact_mismatches=control_mismatch, cases=[c[0] for c in cases],
            matrices_file=str(matrices_path.relative_to(ROOT)),
            matrix_keys=list(matrices), clock_source="CLI assumption, not measured",
            timing_source="Blocking host wall-clock trace replay divided by trace_repeats",
            source_sha256=hashes_before,
            sources_unchanged=hashes_before == source_hashes(),
            warning=(
                "Roundtrip only: no aliasing, matmul consumer, or padding canary. "
                "Custom4 retains four tile closes and programs MOP/stride every batch; replay loads once. "
                "Timing includes both copy stages/configuration and initial/final 128-tile DRAM IO. "
                "Nominal L1 payload GB/s is not physical traffic or isolated pack speed."
            ),
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("RESULT", json.dumps(record), flush=True)
        assert record["sources_unchanged"], "Source changed during qualification"
        assert correctness_pass, "Padded/compact roundtrip correctness failed; matrices saved"
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
