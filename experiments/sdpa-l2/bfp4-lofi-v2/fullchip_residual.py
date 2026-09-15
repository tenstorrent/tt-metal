# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Private all-Q residual LoFi attention: Q256/K512/D128, FP32 DST, chain KV.

Device preprocessing is Q7 (optional prescale) plus fused two-component K/V.
Attention uses cheap exp and no separate P-rounding pass. All reported attention
TFLOPs use the original attention work, not the doubled residual matmul work.
No production/frozen defaults are changed. There is no non-streaming fallback.
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
DIRECTORY = HERE / "fullchip_residual"
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/fullchip_residual/"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


REPRO = module("residual_fullchip_repro", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")
QPREP = module("residual_fullchip_qprep", HERE / "q_prescale.py")
RPREP = module("residual_fullchip_rprep", HERE / "bfp4_residual_preprocess.py")


def cb_specs(variant):
    residual_format, residual_bytes = (
        (ttnn.bfloat8_b, 1088) if variant == "residual48" else (ttnn.bfloat4_b, 576)
    )
    # Q has two slots; every K0/K1/V0/V1 CB has one slot, matching FP32
    # streaming. CB7 aliases CB6 and therefore adds no allocation here.
    return [
        (0, 64, 2048, ttnn.bfloat16),
        (1, 64, 576, ttnn.bfloat4_b),
        (2, 64, 576, ttnn.bfloat4_b),
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
        (17, 64, residual_bytes, residual_format),
        (18, 64, residual_bytes, residual_format),
    ]


def build(device, args, inputs):
    chunks, jobs_per_head = args.length // 512, args.length // 256
    jobs = args.heads * jobs_per_head
    hardware_grid = device.compute_with_storage_grid_size()
    cores = min(args.cores, jobs, hardware_grid.x * hardware_grid.y)
    assert cores >= args.heads and cores % args.heads == 0, "Need equal integer chain lengths per head"
    chain_length = cores // args.heads
    assert chain_length <= jobs_per_head
    coords = [ttnn.CoreCoord(i % hardware_grid.x, i // hardware_grid.x) for i in range(cores)]
    physical = [device.worker_core_from_logical_core(c) for c in coords]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    specs = cb_specs(args.variant)
    cb_bytes = sum(count * size for _, count, size, _ in specs)
    # Blackhole's dev_mem_map.h defines 1536 KiB physical L1. This is a static
    # impossibility check, NOT a claim that all physical L1 is allocator-free.
    raw_l1_bytes = 1536 * 1024
    assert cb_bytes < raw_l1_bytes
    precheck = dict(
        cb_bytes_per_core=cb_bytes,
        blackhole_physical_l1_bytes=raw_l1_bytes,
        raw_l1_minus_cb_bytes=raw_l1_bytes - cb_bytes,
        allocation_check="Static CB footprint only; runtime allocator must also fit firmware/kernel/reserved regions",
    )
    print("CB_PRECHECK", json.dumps(precheck), flush=True)
    originals = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in inputs]
    tq, iq, qcores = QPREP.build(device, originals[0], 7, "bf16", cores, scale=args.q_prescale)
    second_format = "b8_rne5" if args.variant == "residual48" else "b4"
    tk, ik, kcores = RPREP.build(device, originals[1], components=2, ncores=cores, second_format=second_format)
    tv, iv, vcores = RPREP.build(device, originals[2], components=2, ncores=cores, second_format=second_format)
    tensors = [tq, tk[0], tv[0], tk[1], tv[1]]

    def preprocess():
        iq()
        ik()
        iv()

    preprocess()
    preprocess_mismatches = None
    if args.check_preprocess:
        qexpected = QPREP.MODEL.round_significand(inputs[0].float() * args.q_prescale, 7)
        kexpected = RPREP.component_oracle(inputs[1], 2, second_format)
        vexpected = RPREP.component_oracle(inputs[2], 2, second_format)
        expected = [qexpected, kexpected[0], vexpected[0], kexpected[1], vexpected[1]]
        preprocess_mismatches = []
        for name, tensor, oracle in zip(("Q", "K0", "V0", "K1", "V1"), tensors, expected):
            mismatch = int((ttnn.to_torch(tensor).float() != oracle).sum())
            preprocess_mismatches.append(mismatch)
            print("PREPROCESS_CHECK", name, mismatch, flush=True)
            assert mismatch == 0, f"{name} preprocessing mismatch"
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, args.heads, args.length, 128]), ttnn.bfloat16,
        ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    cbs = []
    for index, count, size, fmt in specs:
        formats = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=size)]
        if index == 6:
            formats.append(ttnn.CBFormatDescriptor(buffer_index=7, data_format=fmt, page_size=size))
        cbs.append(ttnn.CBDescriptor(total_size=count * size, core_ranges=grid, format_descriptors=formats))
    defines = dict(
        EXP_APPROX_MODE="1", STATS_GRANULARITY="4", SUB_EXP_GRANULARITY="4",
        MUL_BCAST_GRANULARITY="4", DHT_GRANULARITY="4", REDUCE_GRANULARITY="2",
        SDPA_FP32_STREAMING="1", SDPA_FP32_STATE="1", SDPA_HIFI2_ROUND="1",
        SDPA_LOFI_RESIDUALS="1", SDPA_LOFI_DENOM="1",
    )
    # Deliberately no SDPA_MATCH_HIFI2 and no SDPA_LOFI_ROUND_P: these are the
    # qualified streaming.py residual / cheap-exp / no-P-round settings.
    if args.skip_residual_reconfig:
        assert args.variant == "residual44", "Only identical component formats may skip reconfiguration"
        defines["SDPA_LOFI_SAME_FORMAT_RESIDUAL"] = "1"
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True,
        dst_full_sync_en=False, math_approx_mode=True
    )
    pd = ttnn._ttnn.program_descriptor
    modes = pd.VectorUnpackToDestMode([pd.UnpackToDestMode.Default] * 64)
    for cb in (5, 7, 8, 9, 12, 13, 14):
        modes[cb] = pd.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    read_args, write_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    counts, assignments = [], []
    for i, core in enumerate(coords):
        head, rank = divmod(i, chain_length)
        quotient, remainder = divmod(jobs_per_head, chain_length)
        count = quotient + (rank < remainder)
        offset = head * jobs_per_head + rank * quotient + min(rank, remainder)
        previous = physical[i - 1] if rank else ttnn.CoreCoord(0, 0)
        following = physical[i + 1] if rank + 1 < chain_length else ttnn.CoreCoord(0, 0)
        next_count = quotient + (rank + 1 < remainder) if rank + 1 < chain_length else 0
        assert 0 <= next_count <= count
        assert offset // jobs_per_head == (offset + count - 1) // jobs_per_head == head
        read_args[core.x][core.y] = [t.buffer_address() for t in tensors] + [
            offset, count, rank, chain_length, previous.x, previous.y, following.x, following.y, next_count
        ]
        write_args[core.x][core.y] = [out.buffer_address(), offset, count]
        compute_args[core.x][core.y] = [count]
        counts.append(count)
        assignments.append(dict(core=[core.x, core.y], head=head, rank=rank, first_job=offset, jobs=count))
    assert sum(counts) == jobs
    reader_cta = [8, chunks, jobs_per_head]
    for tensor in tensors:
        reader_cta += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    scale = 1 / math.sqrt(128) / args.q_prescale
    descriptor = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=value)
                    for i, value in enumerate((0, 0, 1))],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "reader_chain.cpp", core_ranges=grid,
                compile_time_args=reader_cta, runtime_args=read_args,
                defines=[("SDPA_READER_BARRIER_TILES", str(args.read_barrier_tiles))],
                config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "writer.cpp", core_ranges=grid,
                compile_time_args=[8] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=write_args, config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "compute.cpp", core_ranges=grid,
                compile_time_args=[chunks, struct.unpack("I", struct.pack("f", scale))[0]],
                runtime_args=compute_args, defines=list(defines.items()), config=config),
        ],
    )

    def attention():
        ttnn.generic_op(tensors + [out], descriptor)

    def combined():
        preprocess()
        attention()

    element_count = args.heads * args.length * 128
    component_bytes = 2 * (576 + (1088 if second_format == "b8_rne5" else 576))
    info = dict(
        **precheck, actual_cores=cores, chain_length=chain_length, q_jobs=jobs,
        jobs_per_core=counts, assignments=assignments, q_chunk=256, k_chunk=512, dim=128,
        input_slots=1, q_slots=2, fp32_dst=True, fidelity="LoFi", exp_quality="cheap", p_round=False,
        executed_qk_pv_matmul_factor=2, defines=defines, preprocessing_cores=[qcores, kcores, vcores],
        preprocessing_mismatches=preprocess_mismatches,
        cb_specs=[(idx, n, size, str(fmt)) for idx, n, size, fmt in specs],
        resident_dram_tensor_payload_bytes=element_count * 10 + element_count // 1024 * component_bytes,
        input_storage="BF16 Q7; BFP4 K0/V0; " + ("RNE5 BFP8" if second_format == "b8_rne5" else "BFP4") + " K1/V1",
        scheduling_note="Existing residual header also waits for V1 before QK; no new lookahead optimization",
    )
    return originals, tensors, out, attention, preprocess, combined, info


def timed(device, invoke, args):
    if not args.iters:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--variant", choices=("residual48", "residual44"), default="residual48")
    parser.add_argument("--q-prescale", type=float, default=1.0)
    parser.add_argument("--skip-residual-reconfig", action="store_true")
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
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--trace-repeats", type=int, default=1)
    args = parser.parse_args()
    assert Path(args.label).name == args.label, "Use a plain fresh label"
    assert args.length >= 512 and args.length % 512 == 0
    assert args.heads > 0 and args.cores > 0 and args.sample_rows > 0
    assert math.isfinite(args.q_prescale) and args.q_prescale > 0
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0
    assert not args.skip_residual_reconfig or args.variant == "residual44"
    path = DIRECTORY / (args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    inputs = REPRO.make_inputs(args.heads, args.length, args.length, 128, args.seed, args.distribution)
    rows = torch.linspace(0, args.length - 1, min(args.sample_rows, args.length)).long().unique()
    reference = REPRO.reference(inputs[0][..., rows, :], inputs[1], inputs[2])
    device = ttnn.open_device(device_id=0, trace_region_size=16777216 if args.iters else 0)
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
        useful_flops = 4 * args.heads * args.length**2 * 128
        sources = [Path(__file__).resolve(), HERE / "numerics.py", HERE / "streaming/compute_streaming.hpp",
                   ROOT / "experiments/sdpa-l2/hybrid-mixed-v1/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp",
                   *sorted(DIRECTORY.glob("*.cpp"))]
        for name in ("q_prescale", "bfp4_residual_preprocess"):
            sources += [HERE / (name + ".py"), *sorted((HERE / name).glob("*.cpp"))]
        ams, cms = attention_time["median_ms"], combined_time["median_ms"]
        record = dict(
            **vars(args), **info, accuracy=accuracy, sampled_query_rows=rows.tolist(),
            accuracy_scope="All heads and KV; explicit sampled Q rows; all output elements checked finite",
            useful_flops=useful_flops, executed_qk_pv_flops=2 * useful_flops,
            attention=attention_time, preprocessing=preprocessing_time, combined=combined_time,
            attention_tflops=useful_flops / (ams * 1e9) if ams else None,
            combined_tflops=useful_flops / (cms * 1e9) if cms else None,
            trace_equal=True if args.iters else None,
            output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            timing_source="Blocking host wall-clock trace replay, divided by trace_repeats",
            warning="Private noncausal square residual streaming; useful FLOPs exclude extra matmuls and preprocessing",
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("RESULT", json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
