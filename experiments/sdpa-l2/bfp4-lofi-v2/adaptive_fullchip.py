# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated adaptive-BFP4 all-Q attention comparisons.

Native RNE versus E/E-1 or E/E-1/E+1 search, for K8/V4 and K4/V4.
BF16 denominator-only compensation is the default attention mode.
Q256/K512/D128, all existing input slots and chain kernels are unchanged.
CPU oracles/means are used only to check results, never to prepare device inputs.
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


REPRO = module("adaptive_fullchip_repro", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")
PREP = module("adaptive_fullchip_preprocess", HERE / "preprocess.py")
B4_PREP = module("adaptive_fullchip_b4_preprocess", HERE / "bfp4_round.py")
# Qualified independent integer oracle; no residual preprocessing is executed.
ORACLE = module("adaptive_fullchip_quant_oracle", HERE / "bfp4_residual_preprocess.py")
ADAPTIVE = module("adaptive_fullchip_quantizer", HERE / "adaptive_bfp4_round.py")


def format_info(name):
    return (ttnn.bfloat4_b, 576) if name == "b4" else (ttnn.bfloat8_b, 1088)


def centered_output_metrics(actual, reference, original_v):
    """Subtract one FP64 original-V column mean from BOTH sampled outputs.

    This is not gain fitting, nor independent output centering. Constant V
    has an analytically zero reference residual: never divide by numerical
    FP64-reference noise in that case.
    """
    bias = original_v.mean(dim=2, keepdim=True, dtype=torch.float64)
    actual_residual = actual.double() - bias
    reference_residual = reference.double() - bias
    error = actual_residual - reference_residual
    constant_v = torch.equal(original_v, original_v[:, :, :1, :].expand_as(original_v))
    reference_norm = reference_residual.norm()
    return dict(
        scope="All heads and explicitly sampled Q rows; original BF16 V, all KV rows",
        centering="Same FP64 mean(original V, KV dimension) subtracted from actual and original FP64 reference",
        gain_alignment=False,
        l2_pct=None if constant_v or reference_norm == 0 else float(100 * error.norm() / reference_norm),
        relative_error_undefined_reason=(
            "Constant V has analytically zero reference residual"
            if constant_v
            else ("Zero reference residual" if reference_norm == 0 else None)
        ),
        centered_reference_rms=0.0 if constant_v else float(reference_residual.square().mean().sqrt()),
        computed_fp64_reference_residual_rms=float(reference_residual.square().mean().sqrt()),
        centered_actual_rms=float(actual_residual.square().mean().sqrt()),
        absolute_error_rms=float(error.square().mean().sqrt()),
        absolute_error_max=float(error.abs().max()),
        constant_v=constant_v,
    )


def make_inputs(args):
    distribution = args.distribution
    channel = {"channel_outlier_k": 1, "channel_outlier_v": 2}.get(distribution)
    inputs = list(
        REPRO.make_inputs(
            args.heads,
            args.length,
            args.length,
            128,
            args.seed,
            "normal" if channel is not None else distribution,
        )
    )
    if channel is not None:
        # Original BF16 test data construction, not algorithm preprocessing.
        inputs[channel][..., ::16] *= 32
    return inputs


def check_preprocessed(original, tensor, index, fmt, args):
    """Check all decoded input values in bounded CPU chunks, outside timings."""
    actual = ttnn.to_torch(tensor).float().reshape(-1)
    source = original.reshape(-1)
    chunk = 524288  # group-aligned; avoid multi-GB integer/FP64 oracle temporaries
    mismatches = 0
    actual_hash, expected_hash = hashlib.sha256(), hashlib.sha256()
    selected = {"-1": 0, "0": 0, "1": 0}
    selection_differs_fp64 = 0
    for start in range(0, source.numel(), chunk):
        values = source[start : start + chunk]
        if index == 0:
            expected = PREP.MODEL.round_significand(values, 7)
        elif fmt == "b4" and args.b4_search != "native":
            expected, stats = ADAPTIVE.oracle(values, args.b4_search)
            for key in selected:
                selected[key] += stats["selected_counts"][key]
            selection_differs_fp64 += stats["selection_differs_fp64_groups"]
        elif fmt == "b4":
            expected = B4_PREP.host_rne_bfp4(values)
        else:
            expected = ORACLE.native_bfp8_rne5(values)
        observed = actual[start : start + chunk]
        # Exact decoded bits; native BFP does not preserve IEEE signed-zero payload.
        observed_bits = torch.where(observed == 0, 0.0, observed).contiguous().view(torch.int32)
        expected_bits = torch.where(expected == 0, 0.0, expected).contiguous().view(torch.int32)
        count = int((observed_bits != expected_bits).sum())
        if count and mismatches == 0:
            torch.save(
                dict(original=values, actual=observed, expected=expected, flat_offset=start),
                HERE / (args.label + ".preprocess-" + ("Q", "K", "V")[index] + ".failure.pt"),
            )
        mismatches += count
        actual_hash.update(observed_bits.numpy().tobytes())
        expected_hash.update(expected_bits.numpy().tobytes())
    record = dict(
        input=("Q", "K", "V")[index],
        format=fmt,
        b4_search=args.b4_search if fmt == "b4" else None,
        decoded_bit_mismatches=mismatches,
        values=source.numel(),
        actual_sha256=actual_hash.hexdigest(),
        expected_sha256=expected_hash.hexdigest(),
        hash_contract="Decoded FP32 bits, canonicalized signed zeros; not packed DRAM bytes",
        selected_counts=selected if fmt == "b4" and args.b4_search != "native" else None,
        selection_differs_fp64_groups=selection_differs_fp64 if fmt == "b4" and args.b4_search != "native" else None,
    )
    print("PREPROCESS_CHECK", json.dumps(record), flush=True)
    assert mismatches == 0, f"Actual preprocessor does not match exact oracle: {record}"
    return record


def build(device, args, inputs):
    destination = "fast_bf16" if args.mode == "denom_bf16" else args.mode
    denom_only = args.mode == "denom_bf16"
    fp32 = destination == "fp32"
    fast = destination == "fast_bf16"
    numerator_compensation = fast and not denom_only
    assert not denom_only or fast
    k_format, v_format = args.kv_formats.split("_")
    k_dtype, k_bytes = format_info(k_format)
    v_dtype, v_bytes = format_info(v_format)
    qt, chunks, jobs_per_head = 8, args.length // 512, args.length // 256
    jobs = args.heads * jobs_per_head
    hardware_grid = device.compute_with_storage_grid_size()
    # Do not silently select an invalid partial per-head chain.
    cores = min(args.cores, jobs, hardware_grid.x * hardware_grid.y)
    assert (
        cores >= args.heads and cores % args.heads == 0
    ), f"Actual cores {cores} must be a positive multiple of heads {args.heads}"
    chain_length = cores // args.heads
    assert chain_length <= jobs_per_head
    coords = [ttnn.CoreCoord(i % hardware_grid.x, i // hardware_grid.x) for i in range(cores)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    physical = [device.worker_core_from_logical_core(c) for c in coords]
    originals = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in inputs]
    tensors, preprocessing, preprocessing_checks, preprocessing_stages = [], [], [], []
    formats = ("bf16", k_format, v_format)
    for i, (src, fmt) in enumerate(zip(originals, formats)):
        adaptive = fmt == "b4" and args.b4_search != "native"
        if adaptive:
            tensor, invoke, prep_cores = ADAPTIVE.build(
                device,
                src,
                ncores=args.cores,
                search=args.b4_search,
                output_format="b4",
            )
        elif fmt == "b4":
            # Qualified optimized baseline, including its original batch4/BF16-DST route.
            tensor, invoke, prep_cores = B4_PREP.build(device, src, ncores=args.cores)
        else:
            tensor, invoke, prep_cores = PREP.build(
                device,
                src,
                bits=7 if i == 0 else 5,
                output_format=fmt,
                ncores=args.cores,
                bfp8_pack_precise=False,
            )
        invoke()
        if args.check_preprocess:
            preprocessing_checks.append(check_preprocessed(inputs[i], tensor, i, fmt, args))
        preprocessing_stages.append(
            dict(
                input=("Q", "K", "V")[i],
                format=fmt,
                actual_cores=prep_cores,
                algorithm=(
                    ("adaptive-" + args.b4_search)
                    if adaptive
                    else ("native-group-RNE" if fmt == "b4" else ("Q7" if i == 0 else "RNE5-B8"))
                ),
                fp32_dst=adaptive or fmt != "b4",
                batch=1 if adaptive else 4,
            )
        )
        tensors.append(tensor)
        preprocessing.append(invoke)

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, args.heads, args.length, 128]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    slots = 1 if fp32 else 2
    state_format, state_bytes = (ttnn.float32, 4096) if fp32 else (ttnn.bfloat16, 2048)
    specs = [
        (0, 2 * qt * 4, 2048, ttnn.bfloat16),
        (1, 64 * slots, k_bytes, k_dtype),
        (2, 64 * slots, v_bytes, v_dtype),
        (3, 1, 2048, ttnn.bfloat16),
        (4, 1, 2048, ttnn.bfloat16),
        (5, 1, state_bytes, state_format),
        (6, qt * 16, state_bytes, state_format),
        (8, qt * 4 * (2 if numerator_compensation else 1), state_bytes, state_format),
        (9, qt * 4 * (2 if numerator_compensation else 1), state_bytes, state_format),
        (10, qt, 2048, ttnn.bfloat16),
        (11, qt, 2048, ttnn.bfloat16),
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
        input_audit.append(
            dict(
                input=("Q", "K", "V")[index],
                cb=index,
                tensor_dtype=str(tensors[index].dtype),
                cb_dtype=str(cb_dtype),
                page_bytes=cb_size,
                capacity_tiles=count,
                slot_tiles=32 if index == 0 else 64,
            )
        )
    cb_bytes = sum(count * size for _, count, size, _ in specs)
    assert cb_bytes < 1536 * 1024, "CBs alone exceed raw Blackhole worker L1"
    cb_audit = [
        dict(
            cb=index,
            tiles=count,
            page_bytes=size,
            dtype=str(fmt),
            total_bytes=count * size,
            aliases=[7] if index == 6 and fp32 else [],
        )
        for index, count, size, fmt in specs
    ]
    print(
        "CB_AUDIT",
        json.dumps(
            dict(
                inputs=input_audit,
                cbs=cb_audit,
                cb_bytes_per_core=cb_bytes,
                raw_l1_headroom_bytes=1536 * 1024 - cb_bytes,
                warning="Raw L1 headroom excludes firmware, program, semaphores and allocator reservations",
            )
        ),
        flush=True,
    )
    cbs = []
    for index, count, size, fmt in specs:
        descriptors = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=size)]
        if index == 6 and fp32:
            descriptors.append(ttnn.CBFormatDescriptor(buffer_index=7, data_format=fmt, page_size=size))
        cbs.append(ttnn.CBDescriptor(total_size=count * size, core_ranges=grid, format_descriptors=descriptors))

    defines = dict(
        EXP_APPROX_MODE="1",
        STATS_GRANULARITY="4" if fp32 else "8",
        SUB_EXP_GRANULARITY="4" if fp32 else "8",
        MUL_BCAST_GRANULARITY="4" if fp32 else "8",
        DHT_GRANULARITY="4",
        REDUCE_GRANULARITY="2" if fp32 else "4",
    )
    if destination == "main_bf16":
        defines["RESIDENT_MAIN"] = "1"
    if fast:
        defines.update(
            SDPA_STREAMING_ACCURACY="1",
            SDPA_STREAMING_NUMERATOR_COMPENSATION="1",
            SDPA_OUT_A_CB="8",
            SDPA_OUT_B_CB="9",
            SDPA_LOFI_FIX_CORRECTION="1",
        )
        if denom_only:
            defines.pop("SDPA_STREAMING_NUMERATOR_COMPENSATION")
    if fp32:
        defines.update(
            SDPA_FP32_STREAMING="1",
            SDPA_FP32_STATE="1",
            SDPA_HIFI2_ROUND="1",
            SDPA_MATCH_HIFI2="1",
        )
    else:
        # Keep recurrence/rescaling multiplication faithful at alpha=1. The
        # two attention matmuls remain LoFi; this matches qualified controls.
        defines["SDPA_LOFI_SAFE_RESCALE"] = "1"
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=fp32,
        dst_full_sync_en=False,
        math_approx_mode=True,
    )
    if fp32:
        pd = ttnn._ttnn.program_descriptor
        modes = pd.VectorUnpackToDestMode([pd.UnpackToDestMode.Default] * 64)
        for cb in (5, 7, 8, 9, 12, 13, 14):
            modes[cb] = pd.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
    read_args, write_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    semaphores = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=value) for i, value in enumerate((0, 0, 1))
    ]
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
    desc = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=semaphores,
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "reader_chain.cpp",
                core_ranges=grid,
                compile_time_args=reader_cta,
                runtime_args=read_args,
                defines=reader_defines,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "writer.cpp",
                core_ranges=grid,
                compile_time_args=[qt] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=write_args,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "compute.cpp",
                core_ranges=grid,
                compile_time_args=[chunks, struct.unpack("I", struct.pack("f", 1 / math.sqrt(128)))[0], qt],
                runtime_args=compute_args,
                defines=list(defines.items()),
                config=config,
            ),
        ],
    )

    def attention():
        ttnn.generic_op(tensors + [out], desc)

    def preprocess():
        for invoke in preprocessing:
            invoke()

    def combined():
        preprocess()
        attention()

    info = dict(
        destination=destination,
        denom_only=denom_only,
        actual_cores=cores,
        q_jobs=jobs,
        jobs_per_core=counts,
        chain_length=chain_length,
        assignments=assignments,
        input_slots=slots,
        cb_bytes_per_core=cb_bytes,
        cb_audit=cb_audit,
        input_audit=input_audit,
        raw_l1_headroom_bytes=1536 * 1024 - cb_bytes,
        defines=defines,
        fidelity="LoFi",
        fp32_dst=fp32,
        q_chunk=256,
        k_chunk=512,
        head_dim=128,
        k_format=k_format,
        v_format=v_format,
        q_preprocessing="Per-value RNE7; BF16 storage",
        k_preprocessing=("BFP4 " + args.b4_search) if k_format == "b4" else "Per-value RNE5 then native BFP8 RNA",
        v_preprocessing=("BFP4 " + args.b4_search) if v_format == "b4" else "Per-value RNE5 then native BFP8 RNA",
        device_preprocessing=True,
        preprocessing_checks=preprocessing_checks,
        preprocessing_stages=preprocessing_stages,
        cpu_input_transform=False,
        copied_attention_host_source_sha256="3d2eba8a39b48e54e18d37055740caaaf2b7ab6635e4489091c8c6da034f294a",
        bfp8_pack_precise=False,
        fix_correction=fast,
        safe_rescale=not fp32,
        reader="Per-head KV chain; source-linear K; K published before V reservation",
        executed_matmul_factor=1,
        output_dtype="BF16",
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
    destination = "fast_bf16" if destination == "denom_bf16" else destination
    sources = [
        Path(__file__).resolve(),
        HERE / "adaptive_bfp4_round.py",
        HERE / "adaptive_bfp4_round/compute.cpp",
        ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py",
        HERE.parent / "bfp4-lofi-v1/probe.py",
        HERE.parent / "bfp4-lofi-v1/numerics.py",
        HERE.parent / "frontier-accuracy-v1/run.py",
        HERE / "preprocess.py",
        HERE / "numerics.py",
        HERE / "bfp4_round.py",
        HERE / "bfp4_residual_preprocess.py",
        HERE / "safe_rescale.hpp",
        HERE / "fast_correction.hpp",
        HERE / "exp_refiner.hpp",
        *(HERE / "preprocess").glob("*.cpp"),
        *(HERE / "bfp4_round").glob("*.cpp"),
        *(HERE / "fullchip").glob("*.cpp"),
    ]
    selected = {
        "main_bf16": "single-core-resident-v1/main",
        "fast_bf16": "bf16-denom-pair-v3/candidate",
        "fp32": "hybrid-mixed-v1/candidate",
    }[destination]
    headers = (
        ROOT / "experiments/sdpa-l2" / selected / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute"
    )
    sources += [
        headers / "compute_common.hpp",
        HERE / "streaming/compute_streaming.hpp" if destination == "fp32" else headers / "compute_streaming.hpp",
    ]
    sources.append(ROOT / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp")
    return sorted(set(sources))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=("denom_bf16", "main_bf16", "fast_bf16", "fp32"), default="denom_bf16")
    parser.add_argument("--b4-search", choices=("native", "minus", "pm"), default="native")
    parser.add_argument("--kv-formats", choices=("b8_b4", "b4_b4"), default="b8_b4")
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--cores", type=int, default=110)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument(
        "--distribution",
        choices=(
            "normal",
            "outliers",
            "scaled_qk",
            "biased_v",
            "common_q",
            "common_k",
            "common_v",
            "uniform",
            "constant_v",
            "uniform_constant_v",
            "channel_outlier_k",
            "channel_outlier_v",
        ),
        default="normal",
    )
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--check-preprocess", action=argparse.BooleanOptionalAction, default=True)
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
    torch.set_num_threads(4)
    pins = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files(args.mode)}
    inputs = make_inputs(args)
    rows = torch.linspace(0, args.length - 1, min(args.sample_rows, args.length)).long().unique()
    reference = REPRO.reference(inputs[0][..., rows, :], inputs[1], inputs[2])
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        originals, tensors, out, attention, preprocess, combined, info = build(device, args, inputs)
        attention()
        actual = ttnn.to_torch(out)
        if not torch.isfinite(actual).all():
            torch.save(dict(inputs=inputs, actual=actual), HERE / (args.label + ".nonfinite.failure.pt"))
            raise AssertionError("Nonfinite attention output")
        accuracy = REPRO.metrics(actual[..., rows, :], reference)
        centered_accuracy = centered_output_metrics(actual[..., rows, :], reference, inputs[2])
        print("ACCURACY", json.dumps(accuracy), flush=True)
        print("CENTERED_OUTPUT_ACCURACY", json.dumps(centered_accuracy), flush=True)
        if args.max_l2 is not None and accuracy["l2_pct"] >= args.max_l2:
            torch.save(
                dict(actual=actual[..., rows, :], reference=reference, rows=rows),
                HERE / (args.label + ".accuracy.failure.pt"),
            )
            raise AssertionError("Accuracy gate failed; do not time this candidate")
        attention_time = timed(device, attention, args)
        preprocessing_time = timed(device, preprocess, args)
        combined_time = timed(device, combined, args)
        assert torch.equal(actual, ttnn.to_torch(out)), "Trace replay changed output"
        flops = 4 * args.heads * args.length**2 * 128
        attention_ms, combined_ms = attention_time["median_ms"], combined_time["median_ms"]
        after_pins = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files(args.mode)
        }
        assert after_pins == pins, "Sources changed during qualification"
        record = dict(
            **vars(args),
            **info,
            accuracy=accuracy,
            centered_output_accuracy=centered_accuracy,
            sampled_query_rows=rows.tolist(),
            accuracy_scope="Original BF16 Q/K/V FP64 reference; all heads and KV, explicit sampled Q rows; all output finite",
            useful_flops=flops,
            attention=attention_time,
            preprocessing=preprocessing_time,
            combined=combined_time,
            attention_tflops=flops / (attention_ms * 1e9) if attention_ms else None,
            combined_tflops=flops / (combined_ms * 1e9) if combined_ms else None,
            trace_equal=True if args.iters else None,
            output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
            source_sha256=pins,
            warning="Private common chain reader, not production SDPA dispatch; useful FLOPs exclude preprocessing; CPU oracle/reference/mean excluded from device path and timings",
            b4_control_contract="native uses qualified batch4 BF16-DST preprocessor; adaptive uses private batch1 FP32-DST search; attention kernels and input buffers identical within a format/mode",
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("RESULT", json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
