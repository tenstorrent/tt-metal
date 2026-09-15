# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validate native BFP payloads in padded L1 pages against compact-page control.

BF16 input -> scalar BFP pack -> BFP unpack/copy -> normal BF16 output.
Both cases use eight resident, tile-distinct source tiles. Default exact inputs
are representable in both BFP4 and BFP8. Full input/output matrices are saved.

This does NOT alias the input and intermediate storage. Passing establishes
padded-page roundtrip addressing, not safe attention alias scheduling or the
number of physical memory transactions. Optional timing measures the complete
copy/pack/unpack/copy/pack pipeline, not attention FLOPs or packer throughput.
"""

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
DIRECTORY = HERE / "padded_bfp_roundtrip"
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/padded_bfp_roundtrip/"
spec = importlib.util.spec_from_file_location("padded_pack_inputs", HERE / "pack_resident.py")
INPUTS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(INPUTS)


def build(device, src, args, page_bytes):
    batch = 4 if args.fp32_dst else 8
    dtype = ttnn.bfloat8_b if args.format == "b8" else ttnn.bfloat4_b
    out = ttnn.allocate_tensor_on_device(src.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cbs = [
        ttnn.CBDescriptor(
            total_size=tiles * size,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=size)],
        )
        for index, tiles, fmt, size in (
            (0, 8, ttnn.bfloat16, 2048),
            (8, 2 * batch, dtype, page_bytes),
            (16, 2 * batch, ttnn.bfloat16, 2048),
        )
    ]
    reader, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader[0][0] = [src.buffer_address()]
    writer[0][0] = [out.buffer_address()]
    cta = [batch, args.tiles]
    descriptor = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=[],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "reader.cpp",
                core_ranges=grid,
                compile_time_args=cta + ttnn.TensorAccessorArgs(src).get_compile_time_args(),
                runtime_args=reader,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "writer.cpp",
                core_ranges=grid,
                compile_time_args=cta + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=writer,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "compute.cpp",
                core_ranges=grid,
                compile_time_args=cta,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    fp32_dest_acc_en=args.fp32_dst,
                    dst_full_sync_en=False,
                    math_approx_mode=False,
                ),
            ),
        ],
    )
    return out, lambda: ttnn.generic_op([src, out], descriptor), 8 * 2048 + 2 * batch * (page_bytes + 2048)


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
    parser.add_argument("--format", choices=("b8", "b4"), default="b8")
    parser.add_argument("--page-bytes", type=int, choices=(2048, 4096), default=2048)
    parser.add_argument("--fp32-dst", action="store_true")
    parser.add_argument("--distribution", choices=("exact", "normal", "positive"), default="exact")
    parser.add_argument("--tiles", type=int, default=8, help="Total processed tiles; use 8192 or more for timing")
    parser.add_argument("--seed", type=int, default=1250)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-repeats", type=int, default=10)
    parser.add_argument("--clock-mhz", type=float, default=1200, help="Assumed clock; not queried/measured")
    args = parser.parse_args()
    assert Path(args.label).name == args.label, "Use a plain fresh label"
    assert args.tiles >= 8 and args.tiles % 8 == 0
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0 and args.clock_mhz > 0
    path, matrices_path = DIRECTORY / (args.label + ".json"), DIRECTORY / (args.label + ".pt")
    assert not path.exists() and not matrices_path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    x = INPUTS.make_input(args.distribution, args.seed)
    native_bytes = 1088 if args.format == "b8" else 576
    device = ttnn.open_device(device_id=0, trace_region_size=4194304 if args.iters else 0)
    try:
        src = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
        cases, matrices = [], {"input_bf16": x}
        for name, page_bytes in (("compact", native_bytes), ("padded", args.page_bytes)):
            out, invoke, cb_bytes = build(device, src, args, page_bytes)
            invoke()
            actual = ttnn.to_torch(out)
            matrices[name + "_output_bf16"] = actual
            finite = bool(torch.isfinite(actual).all())
            mismatch = int((actual != x).sum())
            error = actual.double() - x.double()
            case = dict(
                name=name,
                page_bytes=page_bytes,
                native_payload_bytes=native_bytes,
                cb_bytes_per_core=cb_bytes,
                finite=finite,
                mismatch_vs_input=mismatch,
                max_absolute_error=float(error.abs().max()) if finite else None,
                quantization_l2_pct=float(100 * error.norm() / x.double().norm()) if finite else None,
                output_sha256=hashlib.sha256(actual.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest(),
            )
            cases.append((case, out, invoke))
            print("ROUNDTRIP_CHECK", json.dumps(case), flush=True)
        compact, padded = matrices["compact_output_bf16"], matrices["padded_output_bf16"]
        control_mismatch = int((compact != padded).sum())
        exact_required = args.distribution == "exact"
        correctness_pass = all(c[0]["finite"] for c in cases) and control_mismatch == 0
        correctness_pass &= not exact_required or all(c[0]["mismatch_vs_input"] == 0 for c in cases)
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
            **vars(args),
            cores=1,
            batch=4 if args.fp32_dst else 8,
            pack_width=1,
            input_shape=list(x.shape),
            output_shape=list(compact.shape),
            resident_input_tiles=8,
            saved_output_tiles=8,
            exact_required=exact_required,
            correctness_pass=bool(correctness_pass),
            padded_vs_compact_mismatches=control_mismatch,
            cases=[c[0] for c in cases],
            matrices_file=str(matrices_path.relative_to(ROOT)),
            matrix_keys=list(matrices),
            clock_source="CLI assumption, not measured",
            timing_source="Blocking host wall-clock trace replay divided by trace_repeats",
            source_sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__).resolve(), HERE / "pack_resident.py", *sorted(DIRECTORY.glob("*.cpp"))]
            },
            warning=(
                "Padded-page copy/unpack roundtrip only: no storage aliasing, matmul consumer, or padding canary. "
                "Payload GB/s is nominal accounting, not measured physical L1 traffic. "
                "Each invocation reads eight initial source tiles and saves eight final output tiles."
            ),
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("RESULT", json.dumps(record), flush=True)
        assert correctness_pass, "Padded/compact roundtrip correctness failed; matrices saved"
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
