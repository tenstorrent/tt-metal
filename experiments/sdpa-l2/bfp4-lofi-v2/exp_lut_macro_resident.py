# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One-core Q256/K512 resident LoFi: native exp vs native-grid LUT refinement.

Each invocation loads resident input slots once, repeats identical Q/K/V data
without recurring input DM, and writes only the final Q block. Preprocessing
is device Q-RNE7/BF16 and K/V-RNE5/native-BFP8, identical to exp_lut_macro_streaming.py.
Uses FP32 P/DST/recurrent state and unchanged shared streaming compute.
The optional two-segment FP16 LUT follows the native8-bit exp grid.
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

import exp_lut_macro_streaming as EXP_LUT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/exp_lut_macro_streaming/"


def source_files():
    candidate = ROOT / "experiments/sdpa-l2/hybrid-mixed-v1/candidate"
    return [
        Path(__file__).resolve(),
        HERE / "exp_lut_macro_streaming.py",
        HERE / "preprocess.py",
        HERE / "numerics.py",
        HERE / "bfp4_residual_preprocess.py",
        HERE / "exp_native.hpp",
        HERE / "exp_lut.hpp",
        HERE / "exp_lut_macro.hpp",
        HERE / "exp_lut_models.py",
        ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py",
        HERE.parent / "bfp4-lofi-v1/probe.py",
        HERE.parent / "bfp4-lofi-v1/numerics.py",
        HERE.parent / "frontier-accuracy-v1/run.py",
        ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h",
        HERE / "streaming/compute_streaming.hpp",
        HERE / "exp_lut_macro_streaming/compute_resident.cpp",
        HERE / "exp_lut_macro_streaming/reader_resident.cpp",
        HERE / "exp_lut_macro_streaming/writer_resident.cpp",
        HERE / "resident/reader.cpp",
        HERE / "resident/writer.cpp",
        *sorted((HERE / "preprocess").glob("*.cpp")),
        candidate / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp",
        candidate / "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h",
    ]


def build(device, inputs, q_repeats=16, k_chunks=512, lut_exp=False, check_preprocess=True, raw_lut=False):
    """Return (output, invoke, info); all live device tensors retained by invoke."""
    assert q_repeats > 0 and k_chunks > 0
    assert not raw_lut or lut_exp, "raw_lut requires lut_exp"
    assert tuple(inputs[0].shape) == (1, 1, 256, 128)
    assert all(tuple(x.shape) == (1, 1, 512, 128) for x in inputs[1:])
    assert all(x.dtype == torch.bfloat16 for x in inputs)
    originals = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in inputs]
    tensors, checks, input_hashes = [], [], []
    for i, (src, cpu) in enumerate(zip(originals, inputs)):
        bits, fmt = (7, "bf16") if i == 0 else (5, "b8")
        prepared, prepare, _ = EXP_LUT.PREP.build(device, src, bits, fmt, ncores=1)
        prepare()
        if check_preprocess:
            expected = EXP_LUT.PREP.MODEL.round_significand(cpu, bits)
            if i:
                expected = EXP_LUT.ORACLE.native_bfp8_rne5(cpu)
            actual = ttnn.to_torch(prepared).float()
            mismatch = int((actual != expected).sum())
            assert mismatch == 0, f"QKV preprocessor {i}: {mismatch} mismatches"
            checks.append(mismatch)
            input_hashes.append(hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest())
        tensors.append(prepared)
    out = ttnn.allocate_tensor_on_device(
        [1, 1, 256, 128], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core = ttnn.CoreCoord(0, 0)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    specs = [
        (0, 64, 2048, ttnn.bfloat16),
        (1, 64, 1088, ttnn.bfloat8_b),
        (2, 64, 1088, ttnn.bfloat8_b),
        (3, 1, 2048, ttnn.bfloat16),
        (4, 1, 2048, ttnn.bfloat16),
        (5, 1, 4096, ttnn.float32),
        (6, 128, 4096, ttnn.float32),
        (8, 32, 4096, ttnn.float32),
        (9, 32, 4096, ttnn.float32),
        (10, 8, 2048, ttnn.bfloat16),
        (11, 8, 2048, ttnn.bfloat16),
        (12, 8, 4096, ttnn.float32),
        (13, 8, 4096, ttnn.float32),
        (14, 8, 4096, ttnn.float32),
        (16, 8, 2048, ttnn.bfloat16),
    ]
    cb_bytes = sum(pages * size for _, pages, size, _ in specs)
    assert cb_bytes < 1536 * 1024, "CBs exceed raw Blackhole worker L1"
    print(
        "EXP_LUT_CB_AUDIT",
        json.dumps(
            dict(
                cb_bytes_per_core=cb_bytes,
                raw_l1_headroom_bytes=1536 * 1024 - cb_bytes,
                lut_exp=lut_exp,
                p_format="fp32",
                p_pack_width=4,
                warning="Raw L1 headroom excludes firmware, program, semaphores and allocator reservations",
            )
        ),
        flush=True,
    )
    cbs = []
    for index, pages, page_size, fmt in specs:
        formats = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page_size)]
        if index == 6:
            formats.append(ttnn.CBFormatDescriptor(buffer_index=7, data_format=fmt, page_size=page_size))
        cbs.append(ttnn.CBDescriptor(total_size=pages * page_size, core_ranges=grid, format_descriptors=formats))
    defines = dict(
        EXP_APPROX_MODE="1",
        STATS_GRANULARITY="4",
        SUB_EXP_GRANULARITY="4",
        MUL_BCAST_GRANULARITY="4",
        DHT_GRANULARITY="4",
        REDUCE_GRANULARITY="2",
        SDPA_FP32_STREAMING="1",
        SDPA_FP32_STATE="1",
        SDPA_HIFI2_ROUND="1",
    )
    defines.update(SDPA_LOFI_DENOM="1", SDPA_LOFI_NATIVE_EXP="1")
    if lut_exp:
        defines["SDPA_LOFI_LUT_EXP"] = "1"
        if not raw_lut:
            defines["SDPA_LOFI_LUT_MACRO"] = "1"
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
        math_approx_mode=True,
    )
    pd = ttnn._ttnn.program_descriptor
    modes = pd.VectorUnpackToDestMode([pd.UnpackToDestMode.Default] * 64)
    for cb in (5, 7, 8, 9, 12, 13, 14):
        modes[cb] = pd.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    reader, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader[0][0] = [tensor.buffer_address() for tensor in tensors]
    writer[0][0] = [out.buffer_address()]
    reader_cta = [q_repeats, k_chunks, 1]
    for tensor in tensors:
        reader_cta += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    desc = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=[],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "reader_resident.cpp",
                core_ranges=grid,
                compile_time_args=reader_cta,
                runtime_args=reader,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "writer_resident.cpp",
                core_ranges=grid,
                compile_time_args=[q_repeats] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=writer,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "compute_resident.cpp",
                core_ranges=grid,
                compile_time_args=[q_repeats, k_chunks, struct.unpack("I", struct.pack("f", 1 / math.sqrt(128)))[0]],
                defines=list(defines.items()),
                config=config,
            ),
        ],
    )

    def invoke():
        _keep_originals_alive = originals
        ttnn.generic_op(tensors + [out], desc)

    info = dict(
        cores=1,
        q_chunk=256,
        k_chunk=512,
        head_dim=128,
        input_slots=dict(q=2, k=1, v=1),
        cb_bytes_per_core=sum(pages * page_size for _, pages, page_size, _ in specs),
        cb_specs=[(index, pages, size, str(fmt)) for index, pages, size, fmt in specs],
        defines=defines,
        preprocess_mismatches=checks,
        prepared_input_sha256=input_hashes,
        p_storage="in-place FP32 CB6",
        probability_pack_width=4,
        p_sfpu_prerounding=False,
        exp_fit=(
            "Native8-bit grid plus two-segment FP16 LUT refinement" if lut_exp else "native one-pass approximate exp"
        ),
        native_exp=True,
        exp_lut_refinement=lut_exp,
        exp_refiner="macro8" if lut_exp and not raw_lut else "raw10" if lut_exp else "none",
        exp_coefficients="slopes=0x2f59aee8 intercepts=0x3a1c3c5f; unchanged exp_lut.hpp",
        exp_refiner_issued_instructions_per_two_vectors=8 if lut_exp and not raw_lut else 10 if lut_exp else 0,
        denominator_precision="LoFi P times ones; same effective stored P as PV",
        denominator_matches_pv=True,
        raw_l1_headroom_bytes=1536 * 1024 - cb_bytes,
        accuracy_warning="LUT refinement is experimental; compare measured L2 and gain against identical native control",
        input_preprocess_bfp8_pack_precise=False,
        input_storage="Q RNE7/BF16; K/V RNE5/native-BFP8",
        maxima="BF16",
        recurrence="FP32",
        preprocessing_in_timing=False,
        recurring_input_dm=False,
        invocation_boundary_bytes=64 * 2048 + 2 * 64 * 1088 + 32 * 2048,
    )
    return out, invoke, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--lut-exp", action="store_true", help="Refine native8-bit exp grid using a two-segment FP16 LUT"
    )
    parser.add_argument(
        "--raw-lut", action="store_true", help="With --lut-exp, use unchanged raw LUT instead of macro refinement"
    )
    parser.add_argument("--q-repeats", type=int, default=16)
    parser.add_argument("--k-chunks", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--distribution", default="normal")
    parser.add_argument("--skip-preprocess-check", action="store_true")
    parser.add_argument("--max-l2", type=float, default=10)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trace-repeats", type=int, default=1)
    parser.add_argument(
        "--clock-mhz",
        type=float,
        default=1350,
        help="Assumed device clock for nominal LoFi utilization; not a measured clock",
    )
    args = parser.parse_args()
    assert not args.raw_lut or args.lut_exp, "--raw-lut requires --lut-exp"
    assert args.q_repeats > 0 and args.k_chunks > 0
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0 and args.clock_mhz > 0
    assert Path(args.label).name == args.label
    path = HERE / (args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    inputs = EXP_LUT.REPRO.make_inputs(1, 256, 512, 128, args.seed, args.distribution)
    # Repeating every K/V token equally leaves exact normalized attention
    # unchanged, so this computes all256 Q rows against the resident512 tokens.
    reference = EXP_LUT.REPRO.reference(*inputs)
    files = source_files()
    provenance = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    device = ttnn.open_device(device_id=0, trace_region_size=4194304 if args.iters else 0)
    try:
        out, invoke, info = build(
            device, inputs, args.q_repeats, args.k_chunks, args.lut_exp, not args.skip_preprocess_check, args.raw_lut
        )
        invoke()
        actual = ttnn.to_torch(out)
        assert bool(torch.isfinite(actual).all())
        accuracy = EXP_LUT.REPRO.metrics(actual, reference)
        print("EXP_LUT_RESIDENT_ACCURACY", json.dumps(accuracy), flush=True)
        assert accuracy["l2_pct"] < args.max_l2, "Accuracy gate failed; do not time"
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
            assert torch.equal(actual, ttnn.to_torch(out)), "Replay changed resident output"
        median = statistics.median(times) if times else None
        flops = 4 * 256 * 512 * 128 * args.q_repeats * args.k_chunks
        tflops = flops / (median * 1e9) if median else None
        peak_tflops = 4096 * args.clock_mhz * 1e6 / 1e12
        assert all(
            hashlib.sha256(p.read_bytes()).hexdigest() == provenance[str(p.relative_to(ROOT))] for p in files
        ), "Pinned source changed while this run was active"
        record = dict(
            **vars(args),
            **info,
            accuracy=accuracy,
            reference_scope="all256 query rows; exact repeated-KV equivalence; not distinct262K keys",
            useful_flops=flops,
            median_ms=median,
            replay_ms=times,
            tflops_per_core=tflops,
            nominal_lofi_peak_tflops_per_core=peak_tflops,
            nominal_lofi_utilization_pct=100 * tflops / peak_tflops if tflops else None,
            timing_scope="resident kernel including one initialization read and final Q output write; no recurring input DM",
            trace_equal=True,
            source_sha256=provenance,
            output_sha256=hashlib.sha256(actual.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest(),
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("EXP_LUT_RESIDENT_RESULT", json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
