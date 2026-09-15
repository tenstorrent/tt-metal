# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Independent H16 Q/K + token-axis/adaptive V experiment; full FAST BF16.

New driver only: reuses existing V-axis kernels and preprocessing primitives
without monkey-patching modules or changing qualified sources. H16, V-axis,
adaptive V and grid7 are independent; original BF16 Q/K/V remains the reference.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import struct
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


BASE = module("combined_recipe_vaxis", HERE / "Vtransposed_fullchip.py")
HADAMARD = module("combined_recipe_hadamard", HERE / "hadamard_preprocess.py")
ADAPT = module("combined_recipe_adaptive", HERE / "adaptive_bfp4_round.py")
REPRO, PREP, B4_PREP, ORACLE = BASE.REPRO, BASE.PREP, BASE.B4_PREP, BASE.ORACLE
PREFIX, PRIVATE, COMPUTE = BASE.PREFIX, BASE.PRIVATE, BASE.COMPUTE
format_info, build_transpose = BASE.format_info, BASE.build_transpose
bf16_bits, bf16_bitwise_equal = BASE.bf16_bits, BASE.bf16_bitwise_equal
centered_output_metrics, hashes, tensor_hash = BASE.centered_output_metrics, BASE.hashes, BASE.tensor_hash
timed, verify_trace = BASE.timed, BASE.verify_trace


def transform_metrics(original, actual, block_size):
    """Measure real device transform against FP64; do not require ideal BF16 bits."""
    assert original.dtype == actual.dtype == torch.bfloat16 and original.shape == actual.shape
    a, b = original.reshape(-1, 128), actual.reshape(-1, 128)
    err2 = ref2 = cross = max_abs = 0.0
    rounded_mismatches = 0
    for start in range(0, a.shape[0], 4096):
        reference = HADAMARD.oracle(a[start:start + 4096], block_size).double()
        values = b[start:start + 4096].double()
        assert bool(torch.isfinite(reference).all()) and bool(torch.isfinite(values).all())
        delta = values - reference
        err2 += float(delta.square().sum())
        ref2 += float(reference.square().sum())
        cross += float((values * reference).sum())
        max_abs = max(max_abs, float(delta.abs().max()))
        rounded_mismatches += int((bf16_bits(values.bfloat16()) != bf16_bits(reference.bfloat16())).sum())
    return dict(l2_pct=100 * math.sqrt(err2 / ref2) if ref2 else None,
                gain=cross / ref2 if ref2 else None, max_abs=max_abs,
                bit_mismatches_vs_fp64_transform_rounded_bf16=rounded_mismatches,
                scope="All rows; actual device BF16 spill versus FP64 signed unnormalized H16; measured, not substituted")


def make_inputs(args, distribution):
    allowed = {"normal", "outliers", "channel_v", "outliers_channel_v", "k_outliers_channel_v",
               "scaled_qk", "common_q", "common_k", "common_v", "constant_v"}
    assert distribution in allowed, "Unknown distribution must not silently become normal"
    base = "normal" if distribution in ("channel_v", "k_outliers_channel_v") else (
        "outliers" if distribution == "outliers_channel_v" else distribution)
    inputs = REPRO.make_inputs(args.heads, args.length, args.length, 128, args.seed, base)
    if distribution == "k_outliers_channel_v":
        inputs[1] = REPRO.make_inputs(args.heads, args.length, args.length, 128, args.seed, "outliers")[1]
    if "channel_v" in distribution:
        inputs[2][..., ::16] *= 32
    return inputs


def source_files(destination):
    sources = BASE.source_files(destination) + HADAMARD.source_files()
    sources += [ROOT / path for path in ADAPT.source_pins()]
    sources += [Path(__file__).resolve()]
    return sorted(set(sources))


def build(device, args, inputs):
    assert args.destination == "fast_bf16" and not args.denom_only
    assert args.adaptive_v in ("none", "baseline", "minus", "pm")
    fp32 = False
    fast = args.destination == "fast_bf16"
    numerator_compensation = fast and not args.denom_only
    assert not args.denom_only or fast
    k_format, v_format = args.kv_formats.split("_")
    assert args.adaptive_v == "none" or v_format == "b4", "Adaptive V requires BFP4 V"
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
    prep_sources = list(originals)
    actual_sources = list(inputs)
    rotations, rotation_metadata, rotation_checks = [], [], []
    transform_stages = {}
    if args.h16:
        for i in (0, 1):
            prep_sources[i], rotate, metadata = HADAMARD.build(device, originals[i], 16)
            rotate()
            rotations.append(rotate)
            transform_stages[("q_rotation", "k_rotation")[i]] = rotate
            rotation_metadata.append(dict(input=("Q", "K")[i], **metadata))
            if args.check_preprocess:
                actual_sources[i] = ttnn.to_torch(prep_sources[i]).bfloat16()
                assert bool(torch.isfinite(actual_sources[i]).all()), "Nonfinite device Hadamard output"
                rotation_checks.append(dict(input=("Q", "K")[i],
                                            **transform_metrics(inputs[i], actual_sources[i], 16)))
    transpose_call, transpose_info = None, None
    if args.v_transposed:
        prep_sources[2], transpose_call, transpose_info = build_transpose(device, originals[2], args.cores)
        transpose_call()
        if args.check_preprocess:
            actual_transpose = ttnn.to_torch(prep_sources[2]).bfloat16()
            transpose_mismatch = int((bf16_bits(actual_transpose) != bf16_bits(inputs[2].transpose(-2, -1))).sum())
            assert transpose_mismatch == 0, "BF16 transpose must preserve exact storage bits"
            preprocessing_checks.append(dict(input="V transpose", mismatch=transpose_mismatch,
                                              comparison="BF16 uint16 storage bits, including signed zero"))
            actual_sources[2] = actual_transpose.contiguous()
        transform_stages["v_transpose"] = transpose_call
    for i, (src, fmt) in enumerate(zip(prep_sources, formats)):
        oracle_input = actual_sources[i]
        adaptive = i == 2 and args.adaptive_v != "none"
        adaptive_statistics = None
        if adaptive:
            tensor, invoke, _ = ADAPT.build(device, src, ncores=args.cores, search=args.adaptive_v)
        elif fmt == "b4":
            tensor, invoke, _ = B4_PREP.build(device, src, ncores=args.cores)
        else:
            tensor, invoke, _ = PREP.build(
                device, src, bits=7 if i == 0 else 5,
                output_format=fmt, ncores=args.cores, bfp8_pack_precise=False,
            )
        invoke()
        if args.check_preprocess:
            if i == 0:
                expected = PREP.MODEL.round_significand(oracle_input, 7)
            elif adaptive:
                expected, adaptive_statistics = ADAPT.oracle(oracle_input, args.adaptive_v)
            elif fmt == "b4":
                expected = B4_PREP.host_rne_bfp4(oracle_input)
            else:
                # RNE5 is exact in the packer's E8M6 intermediate. Its shared
                # exponent stage is RNA, not an independently RNE BFP8 codec.
                expected = ORACLE.native_bfp8_rne5(oracle_input)
            actual_input = ttnn.to_torch(tensor).float()
            mismatch = int((actual_input != expected).sum())
            check = dict(input=("Q", "K", "V")[i], format=fmt, mismatch=mismatch,
                         comparison="Exact decoded numeric values; signed zero treated as equal",
                         oracle_source="Actual device BF16 transformed input" if (i < 2 and args.h16) or (i == 2 and args.v_transposed)
                             else "Original BF16 input",
                         adaptive_statistics=adaptive_statistics)
            if i == 2:
                decoded_v = actual_input.transpose(-2, -1).contiguous() if args.v_transposed else actual_input
                check["packed_v_vs_original_bf16"] = REPRO.metrics(decoded_v, inputs[2].double())
                check["packed_v_metric_scope"] = "Decoded storage only, before any further LoFi operand truncation"
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
    if args.grid7_exp:
        defines["SDPA_LOFI_EXP_GRID7"] = "1"
    if args.v_transposed:
        defines["SDPA_V_TRANSPOSED"] = "1"
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
    if args.v_transposed:
        reader_defines.append(("SDPA_V_TRANSPOSED", "1"))
    desc = ttnn.ProgramDescriptor(cbs=cbs, semaphores=semaphores, kernels=[
        ttnn.KernelDescriptor(
            kernel_source=PRIVATE + "reader_chain.cpp", core_ranges=grid,
            compile_time_args=reader_cta, runtime_args=read_args, defines=reader_defines,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=PREFIX + "writer.cpp", core_ranges=grid,
            compile_time_args=[qt] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=write_args, config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=COMPUTE, core_ranges=grid,
            compile_time_args=[chunks, struct.unpack("I", struct.pack("f", 1 / (math.sqrt(128) * (16 if args.h16 else 1))))[0], qt],
            runtime_args=compute_args, defines=list(defines.items()), config=config,
        ),
    ])

    def attention():
        ttnn.generic_op(tensors + [out], desc)

    def quantize():
        for invoke in preprocessing:
            invoke()

    def preprocess():
        for transform in transform_stages.values():
            transform()
        quantize()

    preprocess.stages = dict(transform_stages)
    preprocess.stages.update(zip(("q_quantization", "k_quantization", "v_quantization"), preprocessing))

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
        v_preprocessing=("Adaptive native BFP4 exponent search " + args.adaptive_v) if args.adaptive_v != "none"
            else "Shared-exponent BFP4 RNE" if v_format == "b4" else "Per-value RNE5 then native BFP8 RNA",
        device_preprocessing=True, preprocessing_checks=preprocessing_checks,
        h16=args.h16, qk_rotation_metadata=rotation_metadata, qk_rotation_checks=rotation_checks,
        adaptive_v=args.adaptive_v, adaptive_k=False,
        score_scale=1 / (math.sqrt(128) * (16 if args.h16 else 1)),
        rotation_warning="Signed H16 may amplify common Q/K and inverse channel imbalance; no centering/fallback",
        rotation_reference="Original BF16 Q/K/V; real BF16 transformed spills included, never idealized away",
        grid7_exp=args.grid7_exp, exp_grid_significant_bits=7 if args.grid7_exp else 9,
        bfp8_pack_precise=False, fix_correction=fast, safe_rescale=not fp32,
        reader="Per-head KV chain; K published before V; V^T four source runs scattered N-major" if args.v_transposed
            else "Unchanged ordinary V source-linear control",
        v_group_axis="N: 16 consecutive tokens within one channel" if args.v_transposed else "D: 16 consecutive channels within one token",
        pv_transpose_in1=args.v_transposed, v_transpose=transpose_info,
        v_physical_shape=list(tensors[2].shape), v_cb_tile_order="N-major k_tile*4+d_tile in both controls",
        executed_matmul_factor=1, output_dtype="BF16",
    )
    combined.buffers = (originals, prep_sources, tensors, out, preprocessing, transpose_call, rotations)
    return originals, tensors, out, attention, preprocess, combined, info



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--destination", choices=("fast_bf16",), default="fast_bf16")
    parser.set_defaults(denom_only=False)
    parser.add_argument("--h16", action="store_true", help="Same signed unnormalized device H16 on Q/K; scale divided by16")
    parser.add_argument("--adaptive-v", choices=("none", "baseline", "minus", "pm"), default="none",
                        help="V-only native BFP4 exponent search; K always retains existing quantizer")
    parser.add_argument("--v-transposed", action="store_true", help="Group V along 16 tokens; default ordinary channel-group V")
    parser.add_argument("--grid7-exp", action="store_true", help="Seven-significant-bit direct exp grid; default is unchanged native grid")
    parser.add_argument("--kv-formats", choices=("b8_b8", "b4_b8", "b8_b4", "b4_b4"), default="b4_b4")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--cores", type=int, default=22)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--distributions", "--distribution", nargs="+", default=["normal", "constant_v"])
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--check-preprocess", action="store_true", default=True,
                        help="Always enabled in this qualification driver")
    parser.add_argument("--read-barrier-tiles", type=int, choices=(0, 1, 2, 4, 8, 16, 32, 64), default=2)
    parser.add_argument("--max-l2", type=float)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--trace-repeats", type=int, default=1)
    args = parser.parse_args()
    assert args.length > 0 and args.length % 512 == 0 and args.sample_rows > 0
    assert args.heads > 0 and args.cores >= args.heads and args.cores % args.heads == 0
    assert not args.denom_only or args.destination == "fast_bf16"
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0
    assert Path(args.label).name == args.label
    torch.set_num_threads(4)
    paths = source_files(args.destination)
    pinned = hashes(paths)
    count = args.length if args.length <= 1024 else min(args.sample_rows, args.length)
    rows = torch.linspace(0, args.length - 1, count).long().unique()
    with (HERE / (args.label + ".jsonl")).open("x") as stream:
        def emit(record):
            line = json.dumps(record, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)

        emit(dict(kind="provenance", args=vars(args), source_sha256=pinned,
                  scope="Isolated combined recipe: independent H16 Q/K, V-axis, adaptive-V and grid7 flags; full compensated BF16, Q256/K512, two KV slots; original BF16 reference"))
        for distribution in args.distributions:
            inputs = make_inputs(args, distribution)
            original_hashes = [tensor_hash(x) for x in inputs]
            reference = REPRO.reference(inputs[0][..., rows, :], inputs[1], inputs[2])
            device = ttnn.open_device(device_id=0, trace_region_size=16777216)
            try:
                originals, tensors, out, attention, preprocess, combined, info = build(device, args, inputs)
                combined()
                actual = ttnn.to_torch(out).bfloat16()
                assert bool(torch.isfinite(actual).all()), "Nonfinite output; no timing"
                accuracy = REPRO.metrics(actual[..., rows, :], reference)
                residual = centered_output_metrics(actual[..., rows, :], reference, inputs[2])
                quiet_accuracy = None
                if "channel_v" in distribution:
                    quiet = torch.arange(128) % 16 != 0
                    quiet_accuracy = REPRO.metrics(actual[..., rows, :][..., quiet], reference[..., quiet])
                if args.max_l2 is not None:
                    assert accuracy["l2_pct"] < args.max_l2, "Accuracy gate; no timing"
                verify_trace(device, combined, out, actual)
                timings = {name: timed(device, call, args) for name, call in
                           (("attention", attention), ("preprocessing", preprocess), ("combined", combined))}
                stage_timings = {name: timed(device, call, args) for name, call in preprocess.stages.items()}
                assert bf16_bitwise_equal(actual, ttnn.to_torch(out).bfloat16())
                assert all(bf16_bitwise_equal(ttnn.to_torch(t).bfloat16(), x) for t, x in zip(originals, inputs))
                assert [tensor_hash(x) for x in inputs] == original_hashes
                assert hashes(paths) == pinned, "Pinned sources changed during run"
                flops = 4 * args.heads * args.length ** 2 * 128
                emit(dict(kind="result", distribution=distribution, **vars(args), kernel=info,
                          accuracy=accuracy, centered_output_accuracy=residual,
                          quiet_value_channel_accuracy=quiet_accuracy,
                          bf16_output_rounding_floor=REPRO.metrics(reference.bfloat16(), reference),
                          sampled_query_rows=rows.tolist(), all_query_rows_referenced=len(rows) == args.length,
                          all_output_finite=True, trace_equal=True, original_input_sha256=original_hashes,
                          output_sha256=tensor_hash(actual), useful_flops=flops, **timings,
                          preprocessing_stages=stage_timings,
                          attention_tflops=flops / (timings["attention"]["median_ms"] * 1e9) if args.iters else None,
                          combined_tflops=flops / (timings["combined"]["median_ms"] * 1e9) if args.iters else None,
                          sources_unchanged=True,
                          accuracy_scope="Original BF16 Q/K/V FP64 reference, all KV and explicit Q rows; no gain/reference shifting",
                          timing_scope="Combined includes both real H16 matmuls and real V transpose when enabled, plus Q/K/V quantization and attention; stage timings are a disjoint decomposition of preprocessing, never added twice. Uploads excluded; useful FLOPs exclude preprocessing"))
            finally:
                ttnn.close_device(device)
            del originals, tensors, out, attention, preprocess, combined
            gc.collect()
        emit(dict(kind="complete", sources_unchanged=hashes(paths) == pinned))


if __name__ == "__main__":
    main()

