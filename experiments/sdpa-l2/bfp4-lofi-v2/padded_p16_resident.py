# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Resident-input counterpart of padded_p16_streaming.py, scalar P by default.

Q256/K512/D128, two Q slots and one K/V slot each. Same-pitch BF16 P aliases
FP32 score storage without shrinking inputs. CB7 has its own FIFO counters.
No recurring input DM; preprocessing excluded; only the final Q block saved.
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

import padded_p16_streaming as P16

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/padded_p16_streaming/"


def source_files():
    candidate = ROOT / "experiments/sdpa-l2/hybrid-mixed-v1/candidate"
    return [
        Path(__file__).resolve(), HERE / "padded_p16_streaming.py", HERE / "preprocess.py", HERE / "numerics.py",
        HERE / "bfp4_residual_preprocess.py",
        HERE / "exp_native.hpp",
        HERE / "padded_p16_streaming/compute_streaming.hpp", HERE / "padded_p16_streaming/compute_resident.cpp",
        HERE / "padded_p16_streaming/padded_pack.hpp",
        HERE / "padded_p16_streaming/reader_resident.cpp", HERE / "padded_p16_streaming/writer_resident.cpp",
        HERE / "resident/reader.cpp", HERE / "resident/writer.cpp",
        *sorted((HERE / "preprocess").glob("*.cpp")),
        candidate / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp",
        candidate / "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h",
        ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_common_api.h",
        ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h",
        ROOT / "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h",
        ROOT / "tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h",
        ROOT / "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_template.h",
        ROOT / "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h",
        ROOT / "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_structs.h",
    ]


def build(device, inputs, q_repeats=16, k_chunks=512, p_format="fp32", check_preprocess=True, p_pack_width=1,
          native_exp=False):
    """Return (output, invoke, info); all live device tensors retained by invoke."""
    assert q_repeats > 0 and k_chunks > 0 and p_format in ("fp32", "bf16")
    assert p_pack_width in (1, 4), "Use scalar or opt-in custom width4"
    assert tuple(inputs[0].shape) == (1, 1, 256, 128)
    assert all(tuple(x.shape) == (1, 1, 512, 128) for x in inputs[1:])
    assert all(x.dtype == torch.bfloat16 for x in inputs)
    originals = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in inputs]
    tensors, checks, input_hashes = [], [], []
    for i, (src, cpu) in enumerate(zip(originals, inputs)):
        bits, fmt = (7, "bf16") if i == 0 else (5, "b8")
        prepared, prepare, _ = P16.PREP.build(device, src, bits, fmt, ncores=1)
        prepare()
        if check_preprocess:
            expected = P16.PREP.MODEL.round_significand(cpu, bits)
            if i:
                expected = P16.ORACLE.native_bfp8_rne5(cpu)
            actual = ttnn.to_torch(prepared).float()
            mismatch = int((actual != expected).sum())
            assert mismatch == 0, f"QKV preprocessor {i}: {mismatch} mismatches"
            checks.append(mismatch)
            input_hashes.append(hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest())
        tensors.append(prepared)
    out = ttnn.allocate_tensor_on_device([1, 1, 256, 128], ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                         device, ttnn.DRAM_MEMORY_CONFIG)
    core = ttnn.CoreCoord(0, 0)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    specs = [
        (0, 64, 2048, ttnn.bfloat16),
        (1, 64, 1088, ttnn.bfloat8_b), (2, 64, 1088, ttnn.bfloat8_b),
        (3, 1, 2048, ttnn.bfloat16), (4, 1, 2048, ttnn.bfloat16),
        (5, 1, 4096, ttnn.float32), (6, 128, 4096, ttnn.float32),
        (8, 32, 4096, ttnn.float32), (9, 32, 4096, ttnn.float32),
        (10, 8, 2048, ttnn.bfloat16), (11, 8, 2048, ttnn.bfloat16),
        (12, 8, 4096, ttnn.float32), (13, 8, 4096, ttnn.float32),
        (14, 8, 4096, ttnn.float32), (16, 8, 2048, ttnn.bfloat16),
    ]
    p16 = p_format == "bf16"
    cb_bytes = sum(pages * size for _, pages, size, _ in specs)
    assert cb_bytes < 1536 * 1024, "CBs exceed raw Blackhole worker L1"
    print("P16_CB_AUDIT", json.dumps(dict(
        cb_bytes_per_core=cb_bytes, raw_l1_headroom_bytes=1536 * 1024 - cb_bytes,
        p_format=p_format, p_pack_width=p_pack_width,
        warning="Raw L1 headroom excludes firmware, program, semaphores and allocator reservations",
    )), flush=True)
    cbs = []
    for index, pages, page_size, fmt in specs:
        formats = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page_size)]
        if index == 6:
            assert pages == 128 and page_size == 4096, "Score/P alias must retain identical page pitch and capacity"
            formats.append(ttnn.CBFormatDescriptor(
                buffer_index=7, data_format=ttnn.bfloat16 if p16 else fmt, page_size=page_size,
            ))
        cbs.append(ttnn.CBDescriptor(total_size=pages * page_size, core_ranges=grid, format_descriptors=formats))
    defines = dict(
        EXP_APPROX_MODE="1", STATS_GRANULARITY="4", SUB_EXP_GRANULARITY="4",
        MUL_BCAST_GRANULARITY="4", DHT_GRANULARITY="4", REDUCE_GRANULARITY="2",
        SDPA_FP32_STREAMING="1", SDPA_FP32_STATE="1", SDPA_HIFI2_ROUND="1",
    )
    defines.update(SDPA_LOFI_DENOM="1", SDPA_DIAG_EXP_MODE="1", SDPA_PADDED_P16_PACK_WIDTH=str(p_pack_width))
    if native_exp:
        defines.pop("SDPA_DIAG_EXP_MODE")
        defines["SDPA_LOFI_NATIVE_EXP"] = "1"
    if p16:
        defines["SDPA_PADDED_P16_OUTPUT"] = "1"
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True,
        dst_full_sync_en=False, math_approx_mode=True,
    )
    pd = ttnn._ttnn.program_descriptor
    modes = pd.VectorUnpackToDestMode([pd.UnpackToDestMode.Default] * 64)
    for cb in ((5, 8, 9, 12, 13, 14) if p16 else (5, 7, 8, 9, 12, 13, 14)):
        modes[cb] = pd.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    reader, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader[0][0] = [tensor.buffer_address() for tensor in tensors]
    writer[0][0] = [out.buffer_address()]
    reader_cta = [q_repeats, k_chunks, 1]
    for tensor in tensors:
        reader_cta += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    desc = ttnn.ProgramDescriptor(cbs=cbs, semaphores=[], kernels=[
        ttnn.KernelDescriptor(
            kernel_source=PREFIX + "reader_resident.cpp", core_ranges=grid,
            compile_time_args=reader_cta, runtime_args=reader, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(
            kernel_source=PREFIX + "writer_resident.cpp", core_ranges=grid,
            compile_time_args=[q_repeats] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=writer, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(
            kernel_source=PREFIX + "compute_resident.cpp", core_ranges=grid,
            compile_time_args=[q_repeats, k_chunks, struct.unpack("I", struct.pack("f", 1 / math.sqrt(128)))[0]],
            defines=list(defines.items()), config=config),
    ])

    def invoke():
        _keep_originals_alive = originals
        ttnn.generic_op(tensors + [out], desc)

    info = dict(
        cores=1, q_chunk=256, k_chunk=512, head_dim=128, input_slots=dict(q=2, k=1, v=1),
        cb_bytes_per_core=sum(pages * page_size for _, pages, page_size, _ in specs),
        cb_specs=[(index, pages, size, str(fmt)) for index, pages, size, fmt in specs],
        defines=defines, preprocess_mismatches=checks, prepared_input_sha256=input_hashes,
        p_storage="BF16 CB7 aliases FP32 score CB6; both have 4096-byte pages" if p16 else "in-place FP32 CB6/7",
        probability_pack_width=p_pack_width, p_sfpu_prerounding=False,
        exp_fit="native one-pass approximate exp" if native_exp else "unbiased relative fit",
        denominator="LoFi P times ones; same stored P as PV",
        raw_l1_headroom_bytes=1536 * 1024 - cb_bytes,
        p_page_bytes=4096, p_native_tile_bytes=2048 if p16 else 4096,
        p_alias_dtype="BF16" if p16 else "FP32",
        p_allocation_aliases_score=True, p_fifo_counters_independent=True,
        blocked_p_pack_supported=True,
        p_pack_algorithm="custom Last+output-ADC width4" if p16 and p_pack_width == 4 else "standard LLK",
        p_pack_replay_slots=[17, 31] if p16 and p_pack_width == 4 else None,
        p_pack_replay_reload="After every exp batch" if p16 and p_pack_width == 4 else None,
        reserved_l1_bytes_assumed=111616,
        estimated_allocator_headroom_bytes=1536 * 1024 - cb_bytes - 111616,
        allocation_warning="Reserved-L1 estimate 111616 B from prior device allocation; actual allocator is authoritative",
        input_preprocess_bfp8_pack_precise=False,
        input_storage="Q RNE7/BF16; K/V RNE5/native-BFP8", maxima="BF16", recurrence="FP32",
        preprocessing_in_timing=False, recurring_input_dm=False,
        invocation_boundary_bytes=64 * 2048 + 2 * 64 * 1088 + 32 * 2048,
    )
    return out, invoke, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--p-format", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--native-exp", action="store_true", help="One-pass native exp; default keeps unbiased cubic fit")
    parser.add_argument("--p-pack-width", type=int, choices=(1, 4), default=1,
                        help="1 keeps scalar baseline; 4 selects private padded BF16 MOP or standard FP32 width4")
    parser.add_argument("--q-repeats", type=int, default=16)
    parser.add_argument("--k-chunks", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--distribution", default="normal")
    parser.add_argument("--skip-preprocess-check", action="store_true")
    parser.add_argument("--max-l2", type=float, default=10)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trace-repeats", type=int, default=1)
    parser.add_argument("--clock-mhz", type=float, default=1350,
                        help="Assumed device clock for nominal LoFi utilization; not a measured clock")
    args = parser.parse_args()
    assert args.q_repeats > 0 and args.k_chunks > 0
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0 and args.clock_mhz > 0
    assert Path(args.label).name == args.label
    path = HERE / (args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    inputs = P16.REPRO.make_inputs(1, 256, 512, 128, args.seed, args.distribution)
    # Repeating every K/V token equally leaves exact normalized attention
    # unchanged, so this computes all256 Q rows against the resident512 tokens.
    reference = P16.REPRO.reference(*inputs)
    files = source_files()
    provenance = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    device = ttnn.open_device(device_id=0, trace_region_size=4194304 if args.iters else 0)
    try:
        out, invoke, info = build(device, inputs, args.q_repeats, args.k_chunks,
                                  args.p_format, not args.skip_preprocess_check, args.p_pack_width, args.native_exp)
        invoke()
        actual = ttnn.to_torch(out)
        assert bool(torch.isfinite(actual).all())
        accuracy = P16.REPRO.metrics(actual, reference)
        print("PADDED_P16_RESIDENT_ACCURACY", json.dumps(accuracy), flush=True)
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
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == provenance[str(p.relative_to(ROOT))] for p in files), (
            "Pinned source changed while this run was active"
        )
        record = dict(
            **vars(args), **info, accuracy=accuracy,
            reference_scope="all256 query rows; exact repeated-KV equivalence; not distinct262K keys",
            useful_flops=flops, median_ms=median, replay_ms=times, tflops_per_core=tflops,
            nominal_lofi_peak_tflops_per_core=peak_tflops,
            nominal_lofi_utilization_pct=100 * tflops / peak_tflops if tflops else None,
            timing_scope="resident kernel including one initialization read and final Q output write; no recurring input DM",
            trace_equal=True, source_sha256=provenance,
            output_sha256=hashlib.sha256(actual.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest(),
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("PADDED_P16_RESIDENT_RESULT", json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
