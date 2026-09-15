# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Private LoFi all-Q attention with device signed Hadamard Q/K rotation.

Reuses fullchip/{reader_chain,compute,writer}.cpp unchanged. Q256/K512/D128,
noncausal square attention, one forwarding chain per head. BF16 destinations
retain two K/V slots; FP32 retains one. Q and K get the SAME unnormalized
signed Hadamard, and attention scale is divided by its block size. V is not
rotated. Both device rotations and all quantization are included in combined
timing. No dispatch or frozen-header changes; original BF16 Q/K/V reference.
"""
import argparse
import hashlib
import importlib.util
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
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/fullchip/"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


REPRO = module("rotated_repro", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")
PREP = module("rotated_preprocess", HERE / "preprocess.py")
B4_PREP = module("rotated_b4_preprocess", HERE / "bfp4_round.py")
# Qualified independent integer oracle; no residual preprocessing is executed.
ORACLE = module("rotated_quant_oracle", HERE / "bfp4_residual_preprocess.py")
HADAMARD = module("rotated_hadamard", HERE / "hadamard_preprocess.py")
CENTER = module("rotated_center", HERE / "center_preprocess.py")
MEAN = module("rotated_mean", HERE / "center_mean.py")


def format_info(name):
    return (ttnn.bfloat4_b, 576) if name == "b4" else (ttnn.bfloat8_b, 1088)


def transform_metrics(original, actual, block_size):
    """All-row FP64 transform comparison with bounded reference scratch.

    Different BF16 rounding in the transform is measured, not confused with
    a failure of the subsequent quantizer's exact device-input oracle.
    """
    assert original.dtype == actual.dtype == torch.bfloat16 and original.shape == actual.shape
    a, b = original.reshape(-1, 128), actual.reshape(-1, 128)
    err2, ref2, cross, max_abs, rounded_mismatches = 0.0, 0.0, 0.0, 0.0, 0
    for start in range(0, a.shape[0], 4096):
        source = a[start:start + 4096].reshape(1, 1, -1, 128)
        reference = HADAMARD.oracle(source, block_size).double().reshape(-1, 128)
        values = b[start:start + 4096].double()
        assert bool(torch.isfinite(values).all()) and bool(torch.isfinite(reference).all())
        delta = values - reference
        err2 += float(delta.square().sum())
        ref2 += float(reference.square().sum())
        cross += float((values * reference).sum())
        max_abs = max(max_abs, float(delta.abs().max()))
        rounded_mismatches += int((values.bfloat16() != reference.bfloat16()).sum())
    return dict(l2_pct=100 * math.sqrt(err2 / ref2) if ref2 else None,
        gain=cross / ref2 if ref2 else None, max_abs=max_abs,
        mismatches_vs_fp64_transform_rounded_bf16=rounded_mismatches,
        scope="all rows; FP64 mathematical signed unnormalized Hadamard; no quantization")


def build(device, args, inputs):
    assert args.hadamard_size in (16, 128)
    fp32 = args.destination == "fp32"
    fast = args.destination == "fast_bf16"
    numerator_compensation = fast and not args.denom_only
    assert not args.denom_only or fast
    k_format, v_format = args.kv_formats.split("_")
    k_dtype, k_bytes = format_info(k_format)
    v_dtype, v_bytes = format_info(v_format)
    qt, chunks, jobs_per_head = 8, args.length // 512, args.length // 256
    jobs = args.heads * jobs_per_head
    hardware_grid = device.compute_with_storage_grid_size()
    # Do not silently select an invalid partial per-head chain.
    cores = min(args.cores, jobs, hardware_grid.x * hardware_grid.y)
    assert cores >= args.heads and cores % args.heads == 0, (
        f"Actual cores {cores} must be a positive multiple of heads {args.heads}"
    )
    chain_length = cores // args.heads
    assert chain_length <= jobs_per_head
    coords = [ttnn.CoreCoord(i % hardware_grid.x, i // hardware_grid.x) for i in range(cores)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    physical = [device.worker_core_from_logical_core(c) for c in coords]
    originals = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in inputs]
    tensors, preprocessing, preprocessing_checks = [], [], []
    rotations, rotated_sources, rotation_metadata, rotation_checks = [], [], [], []
    plain_quantizers, mean_calls, center_quantizers = [], [], []
    center_bias, centering_metadata, mean_checks = None, None, []
    formats = ("bf16", k_format, v_format)
    for i, (src, fmt) in enumerate(zip(originals, formats)):
        actual_source = inputs[i]
        if i < 2:
            src, rotate, rotation_info = HADAMARD.build(device, src, args.hadamard_size)
            rotate()
            rotations.append(rotate)
            rotated_sources.append(src)
            rotation_metadata.append(dict(input=("Q", "K")[i], metadata=rotation_info))
            if args.check_preprocess:
                actual_source = ttnn.to_torch(src).bfloat16()
                check = dict(input=("Q", "K")[i], **transform_metrics(inputs[i], actual_source, args.hadamard_size))
                rotation_checks.append(check)
                print("ROTATION_CHECK", json.dumps(check), flush=True)
        centered_k = i == 1 and args.center_k
        if centered_k:
            center_bias, mean_invoke = MEAN.build(device, src, args.center_k_mean_mode)
            mean_invoke()
            mean_calls.append(mean_invoke)
            center_mode = "b4_rne" if fmt == "b4" else "b8_rne5"
            tensor, invoke, center_cores = CENTER.build(device, src, center_bias, args.cores, center_mode)
            center_quantizers.append(invoke)
            centering_metadata = dict(
                input="ACTUAL device rotated BF16 K", mean={**mean_invoke.precision,
                    "input": "ACTUAL device rotated BF16 K"}, actual_center_cores=center_cores,
                mode=center_mode, host_precomputed_bias=False,
                subtraction="Live FP32 SFPU before quantization; no centered BF16 spill",
                invariance="Subtract token-constant rotated-K vector; row-constant logits cancel in softmax; no Q correction",
            )
        elif fmt == "b4":
            tensor, invoke, _ = B4_PREP.build(device, src, ncores=args.cores)
        else:
            tensor, invoke, _ = PREP.build(
                device, src, bits=7 if i == 0 else 5,
                output_format=fmt, ncores=args.cores, bfp8_pack_precise=False,
            )
        if not centered_k:
            plain_quantizers.append(invoke)
        invoke()
        if args.check_preprocess:
            if centered_k:
                actual_bias = ttnn.to_torch(center_bias).bfloat16()
                centered = CENTER.center_oracle(actual_source, actual_bias)
                expected = CENTER.quantize_oracle(centered, center_mode)
                exact_mean = actual_source.double().mean(dim=2, keepdim=True)
                compact_bias = actual_bias[:, :, :1].double()
                mean_check = dict(
                    input="ACTUAL device rotated BF16 K", mode=args.center_k_mean_mode,
                    l2_pct_vs_fp64=CENTER.l2_percent(compact_bias, exact_mean),
                    max_abs_vs_fp64=float((compact_bias - exact_mean).abs().max()),
                    bf16_mismatches_vs_fp64=int((compact_bias.bfloat16() != exact_mean.bfloat16()).sum()),
                    quantization_uses="actual device rounded BF16 mean, not the ideal host mean",
                )
                mean_checks.append(mean_check)
                print("K_MEAN_CHECK", json.dumps(mean_check), flush=True)
            elif i == 0:
                expected = PREP.MODEL.round_significand(actual_source, 7)
            elif fmt == "b4":
                expected = B4_PREP.host_rne_bfp4(actual_source)
            else:
                # RNE5 is exact in the packer's E8M6 intermediate. Its shared
                # exponent stage is RNA, not an independently RNE BFP8 codec.
                expected = ORACLE.native_bfp8_rne5(actual_source)
            actual_input = ttnn.to_torch(tensor).float()
            mismatch = int((actual_input != expected).sum())
            check = dict(input=("Q", "K", "V")[i], format=fmt, mismatch=mismatch,
                         quantizer_reference=("actual device rotated BF16 K minus actual device BF16 mean, FP32 subtraction"
                             if centered_k else "actual device BF16 rotated source" if i < 2 else "original BF16 V"))
            preprocessing_checks.append(check)
            print("PREPROCESS_CHECK", json.dumps(check), flush=True)
            assert mismatch == 0, f"Preprocessing mismatch: {check}"
        tensors.append(tensor)
        preprocessing.append(invoke)

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, args.heads, args.length, 128]), ttnn.bfloat16,
        ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG,
    )
    slots = 1 if fp32 else 2
    state_format, state_bytes = (ttnn.float32, 4096) if fp32 else (ttnn.bfloat16, 2048)
    specs = [
        (0, 2 * qt * 4, 2048, ttnn.bfloat16),
        (1, 64 * slots, k_bytes, k_dtype), (2, 64 * slots, v_bytes, v_dtype),
        (3, 1, 2048, ttnn.bfloat16), (4, 1, 2048, ttnn.bfloat16),
        (5, 1, state_bytes, state_format), (6, qt * 16, state_bytes, state_format),
        (8, qt * 4 * (2 if numerator_compensation else 1), state_bytes, state_format),
        (9, qt * 4 * (2 if numerator_compensation else 1), state_bytes, state_format),
        (10, qt, 2048, ttnn.bfloat16), (11, qt, 2048, ttnn.bfloat16),
        (12, qt * (2 if fast else 1), state_bytes, state_format),
        (13, qt * (2 if fast else 1), state_bytes, state_format),
        (14, qt, state_bytes, state_format),
        (16, 8 if fp32 else 16, 2048, ttnn.bfloat16),
    ]
    # Audit each tensor and input CB independently BEFORE attention can launch.
    # K and V are distinct CB allocations; every chain rank gets these exact
    # same capacities/formats, so forwarding uses equal per-component pointers.
    input_audit = []
    for index, dtype, size in ((0, ttnn.bfloat16, 2048), (1, k_dtype, k_bytes), (2, v_dtype, v_bytes)):
        assert tensors[index].dtype == dtype, f"Input {index} tensor/CB dtype mismatch"
        _, count, cb_size, cb_dtype = specs[index]
        assert cb_size == size and cb_dtype == dtype
        input_audit.append(dict(
            input=("Q", "K", "V")[index], cb=index, tensor_dtype=str(tensors[index].dtype),
            cb_dtype=str(cb_dtype), page_bytes=cb_size, capacity_tiles=count,
            slot_tiles=32 if index == 0 else 64,
        ))
    cb_bytes = sum(count * size for _, count, size, _ in specs)
    assert cb_bytes < 1536 * 1024, "CBs alone exceed raw Blackhole worker L1"
    cb_audit = [dict(cb=index, tiles=count, page_bytes=size, dtype=str(fmt),
                     total_bytes=count * size, aliases=[7] if index == 6 and fp32 else [])
                for index, count, size, fmt in specs]
    print("CB_AUDIT", json.dumps(dict(
        inputs=input_audit, cbs=cb_audit, cb_bytes_per_core=cb_bytes,
        raw_l1_headroom_bytes=1536 * 1024 - cb_bytes,
        warning="Raw L1 headroom excludes firmware, program, semaphores and allocator reservations",
    )), flush=True)
    cbs = []
    for index, count, size, fmt in specs:
        descriptors = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=size)]
        if index == 6 and fp32:
            descriptors.append(ttnn.CBFormatDescriptor(buffer_index=7, data_format=fmt, page_size=size))
        cbs.append(ttnn.CBDescriptor(total_size=count * size, core_ranges=grid, format_descriptors=descriptors))

    defines = dict(
        EXP_APPROX_MODE="1", STATS_GRANULARITY="4" if fp32 else "8",
        SUB_EXP_GRANULARITY="4" if fp32 else "8", MUL_BCAST_GRANULARITY="4" if fp32 else "8",
        DHT_GRANULARITY="4", REDUCE_GRANULARITY="2" if fp32 else "4",
    )
    if args.destination == "main_bf16":
        defines["RESIDENT_MAIN"] = "1"
    if fast:
        defines.update(
            SDPA_STREAMING_ACCURACY="1", SDPA_STREAMING_NUMERATOR_COMPENSATION="1",
            SDPA_OUT_A_CB="8", SDPA_OUT_B_CB="9", SDPA_LOFI_FIX_CORRECTION="1",
        )
        if args.denom_only:
            defines.pop("SDPA_STREAMING_NUMERATOR_COMPENSATION")
    if fp32:
        defines.update(
            SDPA_FP32_STREAMING="1", SDPA_FP32_STATE="1",
            SDPA_HIFI2_ROUND="1", SDPA_MATCH_HIFI2="1",
        )
    else:
        # Keep recurrence/rescaling multiplication faithful at alpha=1. The
        # two attention matmuls remain LoFi; this matches qualified controls.
        defines["SDPA_LOFI_SAFE_RESCALE"] = "1"
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=fp32,
        dst_full_sync_en=False, math_approx_mode=True,
    )
    if fp32:
        pd = ttnn._ttnn.program_descriptor
        modes = pd.VectorUnpackToDestMode([pd.UnpackToDestMode.Default] * 64)
        for cb in (5, 7, 8, 9, 12, 13, 14):
            modes[cb] = pd.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
    read_args, write_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=value)
                  for i, value in enumerate((0, 0, 1))]
    counts, assignments = [], []
    quotient, remainder = divmod(jobs_per_head, chain_length)
    for i, c in enumerate(coords):
        head, rank = divmod(i, chain_length)
        count = quotient + (rank < remainder)
        offset = head * jobs_per_head + rank * quotient + min(rank, remainder)
        prev = physical[i - 1] if rank else ttnn.CoreCoord(0, 0)
        following = physical[i + 1] if rank + 1 < chain_length else ttnn.CoreCoord(0, 0)
        next_count = quotient + (rank + 1 < remainder) if rank + 1 < chain_length else 0
        assert 0 <= next_count <= count and count > 0
        chain_args = [rank, chain_length, prev.x, prev.y, following.x, following.y, next_count]
        read_args[c.x][c.y] = [t.buffer_address() for t in tensors] + [offset, count] + chain_args
        write_args[c.x][c.y] = [out.buffer_address(), offset, count]
        compute_args[c.x][c.y] = [count]
        counts.append(count)
        assignments.append(dict(head=head, rank=rank, first_flat_q_job=offset, jobs=count))
    assert sum(counts) == jobs
    reader_cta = [qt, chunks, jobs_per_head]
    for tensor in tensors:
        reader_cta += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    # Chain reader always publishes K before V reservation and reads source-
    # linear K with an L1 tile-grid scatter. No legacy reader flags are needed.
    reader_defines = [("SDPA_READER_BARRIER_TILES", str(args.read_barrier_tiles))]
    desc = ttnn.ProgramDescriptor(cbs=cbs, semaphores=semaphores, kernels=[
        ttnn.KernelDescriptor(
            kernel_source=PREFIX + "reader_chain.cpp", core_ranges=grid,
            compile_time_args=reader_cta, runtime_args=read_args, defines=reader_defines,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=PREFIX + "writer.cpp", core_ranges=grid,
            compile_time_args=[qt] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=write_args, config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=PREFIX + "compute.cpp", core_ranges=grid,
            compile_time_args=[chunks, struct.unpack("I", struct.pack("f", 1 / (math.sqrt(128) * args.hadamard_size)))[0], qt],
            runtime_args=compute_args, defines=list(defines.items()), config=config,
        ),
    ])

    def attention():
        ttnn.generic_op(tensors + [out], desc)

    def preprocess():
        # Recompute BOTH transforms every invocation, including trace replay.
        # All quantizer closures reference these stable rotated output buffers.
        for rotate in rotations:
            rotate()
        for mean_invoke in mean_calls:
            mean_invoke()
        for invoke in preprocessing:
            invoke()

    def rotate_only():
        for rotate in rotations:
            rotate()

    def quantize_only():
        # With centering enabled, K's fused subtract+quantize is accounted
        # separately. Never run or count that operation twice.
        for invoke in plain_quantizers:
            invoke()

    def mean_only():
        for invoke in mean_calls:
            invoke()

    def center_quantize_only():
        for invoke in center_quantizers:
            invoke()

    preprocess.rotate_only = rotate_only
    preprocess.quantize_only = quantize_only
    preprocess.q_rotate_only, preprocess.k_rotate_only = rotations
    preprocess.mean_only = mean_only
    preprocess.center_quantize_only = center_quantize_only
    preprocess.center_bias = center_bias
    preprocess.rotated_sources = rotated_sources

    def combined():
        preprocess()
        attention()

    info = dict(
        actual_cores=cores, q_jobs=jobs, jobs_per_core=counts, chain_length=chain_length,
        assignments=assignments, input_slots=slots, cb_bytes_per_core=cb_bytes, cb_audit=cb_audit,
        input_audit=input_audit, raw_l1_headroom_bytes=1536 * 1024 - cb_bytes,
        defines=defines, fidelity="LoFi", fp32_dst=fp32, q_chunk=256, k_chunk=512, head_dim=128,
        k_format=k_format, v_format=v_format, q_preprocessing="Per-value RNE7; BF16 storage",
        k_preprocessing="Shared-exponent BFP4 RNE" if k_format == "b4" else "Per-value RNE5 then native BFP8 RNA",
        v_preprocessing="Shared-exponent BFP4 RNE" if v_format == "b4" else "Per-value RNE5 then native BFP8 RNA",
        device_preprocessing=True, preprocessing_checks=preprocessing_checks,
        qk_rotation=rotation_metadata, rotation_checks=rotation_checks,
        k_centering=centering_metadata, k_mean_checks=mean_checks,
        transform="same signed unnormalized Hadamard on Q/K; V unchanged",
        attention_scale=1 / (math.sqrt(128) * args.hadamard_size),
        original_attention_scale=1 / math.sqrt(128),
        ideal_butterfly_transform_flops=2 * args.heads * args.length * 128 * int(math.log2(args.hadamard_size)),
        transform_work_warning="Butterfly count is mathematical only; actual primitive is described in qk_rotation metadata",
        common_mode_warning=("K centering enabled after the real BF16 rotation spill; cannot undo rotation error and does not repair common-Q amplification"
            if args.center_k else "No K centering: orthogonal rotation need not remove coherent common modes and may concentrate some directions"),
        timing_accounting=("Disjoint stages: Q rotation, K rotation, K mean, Q/V quantization, fused K subtraction+quantization. qk_rotations and preprocessing are aggregate controls, not additional summands. Combined independently measures the full chain."
            if args.center_k else "qk_rotations and quantization are disjoint; preprocessing and combined are independently timed aggregates"),
        bfp8_pack_precise=False, fix_correction=fast, safe_rescale=not fp32,
        reader="Per-head KV chain; source-linear K; K published before V reservation",
        executed_matmul_factor=1, output_dtype="BF16",
    )
    return originals, tensors, out, attention, preprocess, combined, info


def timed(device, invoke, args):
    if args.iters == 0:
        return dict(median_ms=None, replay_ms=[])
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(args.trace_repeats):
        invoke()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    times = []
    try:
        for i in range(args.warmup + args.iters):
            start = time.perf_counter()
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            if i >= args.warmup:
                times.append(1000 * (time.perf_counter() - start) / args.trace_repeats)
    finally:
        ttnn.release_trace(device, trace)
    return dict(median_ms=statistics.median(times), replay_ms=times)


def source_files(destination, center_k=False):
    sources = [Path(__file__).resolve(), HERE / "preprocess.py", HERE / "numerics.py",
               HERE / "bfp4_round.py", HERE / "bfp4_residual_preprocess.py",
               HERE / "safe_rescale.hpp", HERE / "fast_correction.hpp", HERE / "exp_refiner.hpp",
               *(HERE / "preprocess").glob("*.cpp"), *(HERE / "bfp4_round").glob("*.cpp"),
               *(HERE / "fullchip").glob("*.cpp")]
    selected = {"main_bf16": "single-core-resident-v1/main", "fast_bf16": "bf16-denom-pair-v3/candidate",
                "fp32": "hybrid-mixed-v1/candidate"}[destination]
    headers = ROOT / "experiments/sdpa-l2" / selected / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute"
    sources += [headers / "compute_common.hpp",
                HERE / "streaming/compute_streaming.hpp" if destination == "fp32" else headers / "compute_streaming.hpp"]
    sources.append(ROOT / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp")
    sources += HADAMARD.source_files()
    if center_k:
        sources += [HERE / "center_mean.py", HERE / "center_preprocess.py",
                    *(HERE / "center_preprocess").glob("*.cpp"),
                    *(HERE / "center_preprocess").glob("*.hpp")]
    sources.append(ROOT / "experiments/sdpa-l2" / selected /
                   "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h")
    return sorted(set(sources))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--destination", choices=("main_bf16", "fast_bf16", "fp32"), default="main_bf16")
    parser.add_argument("--denom-only", action="store_true", help="FAST: compensate denominator only, not output")
    parser.add_argument("--kv-formats", choices=("b8_b8", "b4_b8", "b8_b4", "b4_b4"), default="b8_b8")
    parser.add_argument("--hadamard-size", type=int, choices=(16, 128), default=16)
    parser.add_argument("--center-k", action="store_true",
                        help="Device mean of actual rotated BF16 K, then fused FP32 subtraction and K quantization")
    parser.add_argument("--center-k-mean-mode", choices=("bf16_fpu", "fp32_sfpu"), default="bf16_fpu")
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--cores", type=int, default=110)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--distribution", default="normal")
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--check-preprocess", action="store_true")
    parser.add_argument("--read-barrier-tiles", type=int, choices=(0, 1, 2, 4, 8, 16, 32, 64), default=2)
    parser.add_argument("--max-l2", type=float)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10, help="Zero performs correctness smoke without timing")
    parser.add_argument("--trace-repeats", type=int, default=1)
    args = parser.parse_args()
    assert args.length > 0 and args.length % 512 == 0
    assert args.heads > 0 and args.cores > 0 and args.sample_rows > 0
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0
    assert Path(args.label).name == args.label, "Label must be a filename stem, not a path"
    path = HERE / (args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    inputs = REPRO.make_inputs(args.heads, args.length, args.length, 128, args.seed, args.distribution)
    rows = torch.linspace(0, args.length - 1, min(args.sample_rows, args.length)).long().unique()
    reference = REPRO.reference(inputs[0][..., rows, :], inputs[1], inputs[2])
    sources = source_files(args.destination, args.center_k)
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        originals, tensors, out, attention, preprocess, combined, info = build(device, args, inputs)
        attention()
        actual = ttnn.to_torch(out)
        assert torch.isfinite(actual).all()
        accuracy = REPRO.metrics(actual[..., rows, :], reference)
        print("ACCURACY", json.dumps(accuracy), flush=True)
        if args.max_l2 is not None:
            assert accuracy["l2_pct"] < args.max_l2, "Accuracy gate failed; do not time this candidate"
        attention_time = timed(device, attention, args)
        rotation_time = timed(device, preprocess.rotate_only, args)
        quantization_time = timed(device, preprocess.quantize_only, args)
        centering_times = {}
        if args.center_k:
            centering_times = dict(
                q_rotation=timed(device, preprocess.q_rotate_only, args),
                k_rotation=timed(device, preprocess.k_rotate_only, args),
                k_mean=timed(device, preprocess.mean_only, args),
                k_center_quantization=timed(device, preprocess.center_quantize_only, args),
            )
        preprocessing_time = timed(device, preprocess, args)
        combined_time = timed(device, combined, args)
        assert torch.equal(actual, ttnn.to_torch(out)), "Trace replay changed output"
        flops = 4 * args.heads * args.length**2 * 128
        attention_ms, combined_ms = attention_time["median_ms"], combined_time["median_ms"]
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == source_hashes[str(p.relative_to(ROOT))] for p in sources), (
            "Pinned source changed during the run"
        )
        record = dict(
            **vars(args), **info, accuracy=accuracy, sampled_query_rows=rows.tolist(),
            accuracy_scope="Original BF16 Q/K/V FP64 reference; all heads and KV, explicit sampled Q rows; all output finite",
            useful_flops=flops, attention=attention_time, preprocessing=preprocessing_time, combined=combined_time,
            qk_rotations=rotation_time, quantization=quantization_time,
            **centering_times,
            quantization_scope="Q/V quantization only; fused K centering+quantization is separate" if args.center_k else "Q/K/V quantization",
            attention_tflops=flops / (attention_ms * 1e9) if attention_ms else None,
            combined_tflops=flops / (combined_ms * 1e9) if combined_ms else None,
            trace_equal=True if args.iters else None,
            output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
            source_sha256=source_hashes,
            warning="Private chain reader; useful FLOPs exclude preprocessing arithmetic but combined time includes both Q/K rotations, optional K mean and fused centering, and all quantization",
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("RESULT", json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
