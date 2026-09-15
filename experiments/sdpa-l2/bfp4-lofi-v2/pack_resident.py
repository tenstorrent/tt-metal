# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-core resident BF16 copy -> native pack throughput, not attention FLOPs.

Example correctness controls (each needs a fresh label):
  python experiments/sdpa-l2/bfp4-lofi-v2/pack_resident.py \
      --label pack-b4-exact --output-format b4 --distribution exact --iters 0
  python experiments/sdpa-l2/bfp4-lofi-v2/pack_resident.py \
      --label pack-bf16-f32-w4 --output-format bf16 --fp32-dst --pack-width 4

The trace measurement includes one initial eight-tile DRAM read and one final
batch write per invocation. It measures the copy/unpack/pack/CB pipeline, not
an isolated packer limit. Clock-derived cycles use an explicitly recorded
assumption; this script does not query the hardware clock or device profiler.
Width 4 is supported only for headerless BF16/FP32 outputs. The default LLK
multi-tile MOP does not emit independent BFP exponent sections; BFP uses width 1
explicitly, with no silent scalar fallback labeled as blocked packing.
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
DIRECTORY = HERE / "pack_resident"
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/pack_resident/"


def make_input(distribution, seed):
    shape = (1, 1, 32, 256)  # Eight distinct tiles, in one tile row.
    generator = torch.Generator().manual_seed(seed)
    if distribution == "normal":
        return torch.randn(shape, generator=generator).bfloat16()
    if distribution == "positive":
        return torch.rand(shape, generator=generator).bfloat16()
    # Every native group has an exact BFP4 representation, including a fixed
    # maximum 7/4 to pin the exponent. Tile/row-varying signs and scales expose
    # stale slots, incorrect tiled output addressing, and blocked-pack errors.
    groups = torch.arange(32 * 256 // 16).reshape(-1, 1)
    lanes = torch.arange(16).reshape(1, -1)
    values = ((groups + lanes) % 8).float() / 4
    values *= torch.where((groups + lanes) % 2 == 0, 1.0, -1.0)
    values = torch.ldexp(values, (groups % 9 - 4).int())
    values[::31] = 0
    return values.reshape(shape).bfloat16()


def build(device, src, args):
    bfp_output = args.output_format in ("b4", "b8")
    assert not bfp_output or args.pack_width == 1, "BFP output requires pack width 1"
    batch = 4 if args.fp32_dst else 8
    dtype, tile_bytes = {
        "bf16": (ttnn.bfloat16, 2048),
        "b8": (ttnn.bfloat8_b, 1088),
        "b4": (ttnn.bfloat4_b, 576),
        "fp32": (ttnn.float32, 4096),
    }[args.output_format]
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, batch * 32]), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cbs = [
        ttnn.CBDescriptor(
            total_size=tiles * size,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=size)],
        )
        for index, tiles, fmt, size in ((0, 8, ttnn.bfloat16, 2048), (16, 2 * batch, dtype, tile_bytes))
    ]
    reader, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader[0][0] = [src.buffer_address()]
    writer[0][0] = [out.buffer_address()]
    common_cta = [batch, args.tiles]
    descriptor = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=[],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "reader.cpp",
                core_ranges=grid,
                compile_time_args=common_cta + ttnn.TensorAccessorArgs(src).get_compile_time_args(),
                runtime_args=reader,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "writer.cpp",
                core_ranges=grid,
                compile_time_args=common_cta + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=writer,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "compute.cpp",
                core_ranges=grid,
                compile_time_args=common_cta + [args.pack_width, int(bfp_output)],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    fp32_dest_acc_en=args.fp32_dst,
                    dst_full_sync_en=False,
                    math_approx_mode=False,
                ),
            ),
        ],
    )
    return out, lambda: ttnn.generic_op([src, out], descriptor), batch, tile_bytes


def run(device, args):
    x = make_input(args.distribution, args.seed)
    src = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    out, invoke, batch, tile_bytes = build(device, src, args)
    invoke()
    actual = ttnn.to_torch(out).float()
    # --tiles is a multiple of the eight-tile ring, so the final batch is the
    # rightmost `batch` tiles, not necessarily the first resident input batch.
    expected = x[..., (8 - batch) * 32 :].float()
    assert torch.isfinite(actual).all(), "Nonfinite packed output"
    exact_required = args.distribution == "exact" or args.output_format in ("bf16", "fp32")
    mismatch = int((actual != expected).sum())
    if exact_required:
        assert mismatch == 0, f"{mismatch} mismatches in exact copy/pack control"
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
                    times.append(1000 * (time.perf_counter() - start) / args.trace_repeats)
        finally:
            ttnn.release_trace(device, trace)
        assert torch.equal(actual, ttnn.to_torch(out).float()), "Trace replay changed output"
    median_ms = statistics.median(times) if times else None
    tiles_per_second = args.tiles * 1000 / median_ms if median_ms else None
    error = actual.double() - expected.double()
    return dict(
        **vars(args),
        cores=1,
        resident_tiles=8,
        batch=batch,
        output_cb_slots=2,
        cb_bytes=8 * 2048 + 2 * batch * tile_bytes,
        input_tile_bytes=2048,
        output_tile_bytes=tile_bytes,
        startup_input_dram_bytes=8 * 2048,
        final_output_dram_bytes=batch * tile_bytes,
        steady_state_dram_bytes=0,
        median_ms=median_ms,
        trace_ms_per_invocation=times,
        tiles_per_second=tiles_per_second,
        l1_input_payload_GBps=tiles_per_second * 2048 / 1e9 if tiles_per_second else None,
        l1_output_payload_GBps=tiles_per_second * tile_bytes / 1e9 if tiles_per_second else None,
        l1_read_write_payload_GBps=tiles_per_second * (2048 + tile_bytes) / 1e9 if tiles_per_second else None,
        estimated_cycles_per_tile=args.clock_mhz * 1e6 / tiles_per_second if tiles_per_second else None,
        clock_source="CLI assumption, not measured",
        timing_source="blocking host wall-clock trace replay, divided by trace_repeats",
        exact_required=exact_required,
        mismatch_vs_bf16_input=mismatch,
        quantization_l2_pct=float(100 * error.norm() / expected.double().norm()),
        max_absolute_error=float(error.abs().max()),
        trace_equal=True if times else None,
        final_output_sha256=hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest(),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(), *sorted(DIRECTORY.glob("*.cpp"))]
        },
        warning=(
            "Copy/unpack/native-pack/CB throughput, not packer-only throughput or attention TFLOPs. "
            "L1 payload rates omit internal transactions and do not measure total physical L1 traffic. "
            "BFP random-input quantization is reported without a native-pack exact oracle."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-format", choices=("bf16", "b8", "b4", "fp32"), default="bf16")
    parser.add_argument("--fp32-dst", action="store_true")
    parser.add_argument("--pack-width", type=int, choices=(1, 4), default=1)
    parser.add_argument("--tiles", type=int, default=8192)
    parser.add_argument("--distribution", choices=("normal", "positive", "exact"), default="positive")
    parser.add_argument("--seed", type=int, default=1250)
    parser.add_argument("--clock-mhz", type=float, default=1200, help="Assumed clock, NOT queried/measured")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--trace-repeats", type=int, default=10)
    args = parser.parse_args()
    if args.output_format in ("b4", "b8") and args.pack_width != 1:
        parser.error(
            "BFP output requires --pack-width 1: the default multi-tile LLK MOP closes once per block, "
            "not once per BFP tile, so independent exponent sections are not framed correctly. "
            "No scalar fallback is substituted for this blocked-pack control."
        )
    assert Path(args.label).name == args.label, "Use a plain fresh label"
    assert args.tiles >= 8 and args.tiles % 8 == 0
    assert args.clock_mhz > 0 and args.warmup >= 0 and args.iters >= 0 and args.trace_repeats > 0
    path = DIRECTORY / (args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    device = ttnn.open_device(device_id=args.device_id, trace_region_size=4194304 if args.iters else 0)
    try:
        record = run(device, args)
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("RESULT " + json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
