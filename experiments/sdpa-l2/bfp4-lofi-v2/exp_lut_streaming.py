# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Private FP32-DST/state LoFi attention: native exp vs native-grid LUT refinement.

Fixed Q256/K512/D128, Q7/BF16 and K/V RNE5/native-BFP8, unchanged FP32 P
storage and matched-P LoFi denominator. Optional refinement follows the native
8-bit grid with a two-segment FP16 LUT. The flag-off control uses the original
native exp without LUT calls. Shared compute kernels remain unchanged.
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
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/exp_lut_streaming/"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


REPRO = module("exp_lut_repro", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")
PREP = module("exp_lut_preprocess", HERE / "preprocess.py")
B4_PREP = module("exp_lut_b4_preprocess", HERE / "bfp4_round.py")
# Qualified independent integer oracle; no residual preprocessing is executed.
ORACLE = module("exp_lut_quant_oracle", HERE / "bfp4_residual_preprocess.py")


def format_info(name):
    return (ttnn.bfloat4_b, 576) if name == "b4" else (ttnn.bfloat8_b, 1088)


def build(device, args, inputs):
    assert args.destination == "fp32" and args.kv_formats == "b8_b8"
    lut_exp = args.lut_exp
    fp32 = True
    fast = False
    numerator_compensation = False
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
    formats = ("bf16", k_format, v_format)
    for i, (src, fmt) in enumerate(zip(originals, formats)):
        if fmt == "b4":
            tensor, invoke, _ = B4_PREP.build(device, src, ncores=args.cores)
        else:
            tensor, invoke, _ = PREP.build(
                device, src, bits=7 if i == 0 else 5,
                output_format=fmt, ncores=args.cores, bfp8_pack_precise=False,
            )
        invoke()
        if args.check_preprocess:
            if i == 0:
                expected = PREP.MODEL.round_significand(inputs[i], 7)
            elif fmt == "b4":
                expected = B4_PREP.host_rne_bfp4(inputs[i])
            else:
                # RNE5 is exact in the packer's E8M6 intermediate. Its shared
                # exponent stage is RNA, not an independently RNE BFP8 codec.
                expected = ORACLE.native_bfp8_rne5(inputs[i])
            actual_input = ttnn.to_torch(tensor).float()
            mismatch = int((actual_input != expected).sum())
            check = dict(input=("Q", "K", "V")[i], format=fmt, mismatch=mismatch)
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
    defines.update(SDPA_FP32_STREAMING="1", SDPA_FP32_STATE="1", SDPA_HIFI2_ROUND="1")
    defines.update(SDPA_LOFI_DENOM="1", SDPA_LOFI_NATIVE_EXP="1")
    if lut_exp:
        defines["SDPA_LOFI_LUT_EXP"] = "1"
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
            compile_time_args=[chunks, struct.unpack("I", struct.pack("f", 1 / math.sqrt(128)))[0], qt],
            runtime_args=compute_args, defines=list(defines.items()), config=config,
        ),
    ])

    def attention():
        ttnn.generic_op(tensors + [out], desc)

    def preprocess():
        for invoke in preprocessing:
            invoke()

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
        bfp8_pack_precise=False, fix_correction=fast, safe_rescale=not fp32,
        reader="Per-head KV chain; source-linear K; K published before V reservation",
        executed_matmul_factor=1, output_dtype="BF16",
        p_storage="in-place FP32 CB6/7", probability_pack_width=4,
        exp_fit="Native8-bit grid plus two-segment FP16 LUT refinement" if lut_exp else "native one-pass approximate exp",
        native_exp=True, exp_lut_refinement=lut_exp,
        denominator_precision="LoFi P times ones; same effective stored P as PV",
        denominator_matches_pv=True,
        p_sfpu_prerounding=False,
        accuracy_warning="LUT refinement is experimental; compare measured L2 and gain against identical native control",
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


def source_files(destination):
    sources = [Path(__file__).resolve(), HERE / "preprocess.py", HERE / "numerics.py",
               HERE / "bfp4_round.py", HERE / "bfp4_residual_preprocess.py",
               HERE / "safe_rescale.hpp", HERE / "fast_correction.hpp", HERE / "exp_refiner.hpp", HERE / "exp_native.hpp", HERE / "exp_lut.hpp",
               *(HERE / "preprocess").glob("*.cpp"), *(HERE / "bfp4_round").glob("*.cpp"),
               *(HERE / "fullchip").glob("*.cpp"), *(HERE / "exp_lut_streaming").glob("*.cpp")]
    selected = {"main_bf16": "single-core-resident-v1/main", "fast_bf16": "bf16-denom-pair-v3/candidate",
                "fp32": "hybrid-mixed-v1/candidate"}[destination]
    headers = ROOT / "experiments/sdpa-l2" / selected / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute"
    sources += [headers / "compute_common.hpp",
                HERE / "streaming/compute_streaming.hpp" if destination == "fp32" else headers / "compute_streaming.hpp"]
    sources.append(ROOT / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp")
    sources.append(ROOT / "experiments/sdpa-l2/hybrid-mixed-v1/candidate/tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h")
    return sorted(set(sources))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.set_defaults(destination="fp32", kv_formats="b8_b8")
    parser.add_argument("--lut-exp", action="store_true", help="Refine native8-bit exp grid using a two-segment FP16 LUT")
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--cores", type=int, default=110)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--distribution", default="normal")
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--check-preprocess", action="store_true")
    parser.add_argument("--read-barrier-tiles", type=int, choices=(0, 1, 2, 4, 8, 16, 32, 64), default=2)
    parser.add_argument("--max-l2", type=float, default=10)
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
    sources = source_files(args.destination)
    provenance = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
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
        preprocessing_time = timed(device, preprocess, args)
        combined_time = timed(device, combined, args)
        assert torch.equal(actual, ttnn.to_torch(out)), "Trace replay changed output"
        flops = 4 * args.heads * args.length**2 * 128
        attention_ms, combined_ms = attention_time["median_ms"], combined_time["median_ms"]
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == provenance[str(p.relative_to(ROOT))] for p in sources), (
            "Pinned source changed while this run was active"
        )
        record = dict(
            **vars(args), **info, accuracy=accuracy, sampled_query_rows=rows.tolist(),
            accuracy_scope="Original BF16 Q/K/V FP64 reference; all heads and KV, explicit sampled Q rows; all output finite",
            useful_flops=flops, attention=attention_time, preprocessing=preprocessing_time, combined=combined_time,
            attention_tflops=flops / (attention_ms * 1e9) if attention_ms else None,
            combined_tflops=flops / (combined_ms * 1e9) if combined_ms else None,
            trace_equal=True if args.iters else None,
            output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
            source_sha256=provenance,
            warning="Private common chain reader, not production SDPA dispatch; useful FLOPs exclude preprocessing",
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("RESULT", json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
