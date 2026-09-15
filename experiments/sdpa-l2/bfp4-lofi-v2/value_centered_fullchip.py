# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Private LoFi BF16 attention with optional device V4 centering and output bias.

Reuses fullchip/{reader_chain,compute,writer}.cpp unchanged. Q256/K512/D128,
noncausal square attention, one forwarding chain per head. BF16 destinations
retain two K/V slots. Centered modes use an explicitly costed BF16 epilogue;
the attention core still writes BF16, so there is an extra intermediate
rounding. Original BF16 Q/K/V are retained as the FP64 accuracy reference.
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


REPRO = module("value_centered_repro", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")
PREP = module("value_centered_preprocess", HERE / "preprocess.py")
B4_PREP = module("value_centered_b4_preprocess", HERE / "bfp4_round.py")
# Qualified independent integer oracle; no residual preprocessing is executed.
ORACLE = module("value_centered_quant_oracle", HERE / "bfp4_residual_preprocess.py")
# These modules intentionally import their qualified dependencies normally.
import center_mean as MEAN
import center_preprocess as CENTER


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
        relative_error_undefined_reason="Constant V has analytically zero reference residual" if constant_v else
            ("Zero reference residual" if reference_norm == 0 else None),
        centered_reference_rms=0.0 if constant_v else float(reference_residual.square().mean().sqrt()),
        computed_fp64_reference_residual_rms=float(reference_residual.square().mean().sqrt()),
        centered_actual_rms=float(actual_residual.square().mean().sqrt()),
        absolute_error_rms=float(error.square().mean().sqrt()),
        absolute_error_max=float(error.abs().max()),
        constant_v=constant_v,
    )


def build_centered_value(device, src, args):
    """V4, recompute callable, stable compact BF16 epilogue bias, metadata.

    Both means are generated on device. Use the actual rounded repeated
    biases, not an unrounded FP32 last_mean from the optional SFPU mean mode.
    """
    assert args.center_mode in ("original_mean", "matched_mean")
    original_bias, original_mean = MEAN.build(device, src, args.mean_mode)
    value, quantize, quant_cores = CENTER.build(device, src, original_bias, args.cores, "b4_rne")
    compact_shape = [1, src.shape[1], 1, 128]
    compact_bias = ttnn.allocate_tensor_on_device(compact_shape, ttnn.bfloat16,
        ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    represented = represented_bias = represented_mean = delta_bias = None
    if args.center_mode == "matched_mean":
        represented = ttnn.allocate_tensor_on_device(src.shape, ttnn.bfloat16,
            ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
        represented_bias, represented_mean = MEAN.build(device, represented, args.mean_mode)
        delta_bias = ttnn.allocate_tensor_on_device(original_bias.shape, ttnn.bfloat16,
            ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)

    def invoke():
        original_mean()
        quantize()
        epilogue_source = original_bias
        if represented is not None:
            # Decodes the actual packed BFP4, not a host/requantized proxy.
            ttnn.typecast(value, ttnn.bfloat16, output_tensor=represented)
            represented_mean()
            ttnn.subtract(original_bias, represented_bias, output_tensor=delta_bias,
                          fast_and_approximate_mode=False)
            epilogue_source = delta_bias
        # Logical1 broadcasts across all output rows. Passing repeated logical32
        # directly would not broadcast to an arbitrary sequence length.
        ttnn.slice(epilogue_source, [0, 0, 0, 0], compact_shape, [1, 1, 1, 1],
                   output_tensor=compact_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    invoke.original_bias = original_bias
    invoke.original_mean = original_mean
    invoke.represented = represented
    invoke.represented_bias = represented_bias
    invoke.represented_mean = represented_mean
    invoke.compact_bias = compact_bias
    info = dict(mode=args.center_mode,
        original_mean={**original_mean.precision, "input": "Original device BF16 V"},
        represented_mean={**represented_mean.precision, "input": "Actual packed centered V4 decoded exactly to device BF16"}
            if represented_mean is not None else None,
        centered_value="FP32 SFPU original BF16 V minus actual BF16 mean, native-group RNE/BFP4",
        quantization_cores=quant_cores,
        output_bias="original BF16 mean" if represented is None else "BF16(original BF16 mean - device BF16 mean(decoded V4))",
        represented_v_exact_in_lofi_right_operand=True,
        extra_rounding="Core normalized output BF16, then BF16 broadcast add produces final BF16; matched bias subtraction also BF16",
        epilogue_and_bias_subtract_fast_and_approximate_mode=False,
        extra_full_v_decode_in_preprocessing=represented is not None,
        corrected_unweighted_mean_warning="Matched mode corrects represented unweighted V mean; attention-weighted quantization error remains")
    return value, invoke, compact_bias, info


def check_centered_value(original, value, prepare):
    """Exact packed-value oracle against actual device rounded mean."""
    bias = ttnn.to_torch(prepare.original_bias).bfloat16()
    centered = CENTER.center_oracle(original, bias)
    expected = CENTER.quantize_oracle(centered, "b4_rne")
    actual = ttnn.to_torch(value).float()
    mismatch = int((actual != expected).sum())
    assert mismatch == 0, f"Centered V4 mismatch count {mismatch}"
    compact = ttnn.to_torch(prepare.compact_bias).bfloat16()
    source_mean = original.double().mean(dim=2, keepdim=True)
    represented_mean64 = actual.double().mean(dim=2, keepdim=True)
    actual_bias = bias[:, :, :1].float()
    result = dict(input="V", format="b4", mismatch=mismatch,
        quantizer_reference="FP32 V-minus-ACTUAL device rounded BF16 mean; direct RNE BFP4 oracle",
        original_mean_max_abs_vs_fp64=float((actual_bias.double() - source_mean).abs().max()),
        represented_centered_mean_rms=float(represented_mean64.square().mean().sqrt()),
        effective_represented_plus_bias_mean_max_abs_vs_original=float(
            (represented_mean64 + compact.double() - source_mean).abs().max()))
    if prepare.represented is not None:
        decoded = ttnn.to_torch(prepare.represented).float()
        assert torch.equal(decoded, actual), "V4-to-BF16 decode mismatch"
        represented_bias = ttnn.to_torch(prepare.represented_bias).bfloat16()
        expected_bias = (bias[:, :, :1].float() - represented_bias[:, :, :1].float()).bfloat16()
        assert torch.equal(compact, expected_bias), "Matched bias subtraction/slice differs from BF16 rounding oracle"
        result["represented_mean_max_abs_vs_fp64"] = float(
            (represented_bias[:, :, :1].double() - represented_mean64).abs().max())
    else:
        assert torch.equal(compact, bias[:, :, :1]), "Original mean compact slice mismatch"
    return result


def build(device, args, inputs):
    assert args.destination in ("main_bf16", "fast_bf16")
    assert args.kv_formats in ("b8_b4", "b4_b4")
    assert args.center_mode in ("none", "original_mean", "matched_mean")
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
    value_prepare = epilogue_bias = None
    centering_info = dict(mode="none", extra_rounding="none; original attention BF16 output")
    formats = ("bf16", k_format, v_format)
    for i, (src, fmt) in enumerate(zip(originals, formats)):
        if i == 2 and args.center_mode != "none":
            tensor, invoke, epilogue_bias, centering_info = build_centered_value(device, src, args)
            value_prepare = invoke
        elif fmt == "b4":
            tensor, invoke, _ = B4_PREP.build(device, src, ncores=args.cores)
        else:
            tensor, invoke, _ = PREP.build(
                device, src, bits=7 if i == 0 else 5,
                output_format=fmt, ncores=args.cores, bfp8_pack_precise=False,
            )
        invoke()
        if args.check_preprocess:
            if i == 2 and value_prepare is not None:
                check = check_centered_value(inputs[i], tensor, value_prepare)
                preprocessing_checks.append(check)
                print("PREPROCESS_CHECK", json.dumps(check), flush=True)
            elif i == 0:
                expected = PREP.MODEL.round_significand(inputs[i], 7)
            elif fmt == "b4":
                expected = B4_PREP.host_rne_bfp4(inputs[i])
            else:
                # RNE5 is exact in the packer's E8M6 intermediate. Its shared
                # exponent stage is RNA, not an independently RNE BFP8 codec.
                expected = ORACLE.native_bfp8_rne5(inputs[i])
            if not (i == 2 and value_prepare is not None):
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
    final_out = out if epilogue_bias is None else ttnn.allocate_tensor_on_device(
        out.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
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
            compile_time_args=[chunks, struct.unpack("I", struct.pack("f", 1 / math.sqrt(128)))[0], qt],
            runtime_args=compute_args, defines=list(defines.items()), config=config,
        ),
    ])

    def attention():
        ttnn.generic_op(tensors + [out], desc)

    def epilogue():
        if epilogue_bias is not None:
            ttnn.add(out, epilogue_bias, output_tensor=final_out, fast_and_approximate_mode=False)

    def attention_with_epilogue():
        attention()
        epilogue()

    def preprocess():
        for invoke in preprocessing:
            invoke()

    def combined():
        preprocess()
        attention_with_epilogue()

    attention.core_output = out
    attention.epilogue = epilogue
    attention.with_epilogue = attention_with_epilogue
    attention.epilogue_bias = epilogue_bias
    preprocess.value_prepare = value_prepare

    info = dict(
        actual_cores=cores, q_jobs=jobs, jobs_per_core=counts, chain_length=chain_length,
        assignments=assignments, input_slots=slots, cb_bytes_per_core=cb_bytes, cb_audit=cb_audit,
        input_audit=input_audit, raw_l1_headroom_bytes=1536 * 1024 - cb_bytes,
        defines=defines, fidelity="LoFi", fp32_dst=fp32, q_chunk=256, k_chunk=512, head_dim=128,
        k_format=k_format, v_format=v_format, q_preprocessing="Per-value RNE7; BF16 storage",
        k_preprocessing="Shared-exponent BFP4 RNE" if k_format == "b4" else "Per-value RNE5 then native BFP8 RNA",
        v_preprocessing="Shared-exponent BFP4 RNE" if v_format == "b4" else "Per-value RNE5 then native BFP8 RNA",
        device_preprocessing=True, preprocessing_checks=preprocessing_checks,
        value_centering=centering_info,
        epilogue_dtype="BF16", core_output_dtype="BF16",
        epilogue_materialized=args.center_mode != "none",
        bfp8_pack_precise=False, fix_correction=fast, safe_rescale=not fp32,
        reader="Per-head KV chain; source-linear K; K published before V reservation",
        executed_matmul_factor=1, output_dtype="BF16",
    )
    return originals, tensors, final_out, attention, preprocess, combined, info


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
               HERE / "safe_rescale.hpp", HERE / "fast_correction.hpp", HERE / "exp_refiner.hpp",
               *(HERE / "preprocess").glob("*.cpp"), *(HERE / "bfp4_round").glob("*.cpp"),
               *(HERE / "fullchip").glob("*.cpp")]
    selected = {"main_bf16": "single-core-resident-v1/main", "fast_bf16": "bf16-denom-pair-v3/candidate",
                "fp32": "hybrid-mixed-v1/candidate"}[destination]
    headers = ROOT / "experiments/sdpa-l2" / selected / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute"
    sources += [headers / "compute_common.hpp",
                HERE / "streaming/compute_streaming.hpp" if destination == "fp32" else headers / "compute_streaming.hpp"]
    sources.append(ROOT / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp")
    sources += [HERE / "center_mean.py", HERE / "center_preprocess.py",
                *(HERE / "center_preprocess").glob("*.cpp"), *(HERE / "center_preprocess").glob("*.hpp")]
    sources.append(ROOT / "experiments/sdpa-l2" / selected /
                   "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h")
    sources += [ROOT / "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl",
                ROOT / "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl",
                ROOT / "ttnn/cpp/ttnn/operations/copy/typecast/typecast.cpp",
                ROOT / "ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp",
                ROOT / "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp",
                ROOT / "ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp",
                ROOT / "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp",
                ROOT / "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp",
                ROOT / "ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp",
                ROOT / "ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_h_program_factory.cpp",
                ROOT / "ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp",
                ROOT / "ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_tile.cpp"]
    # Selected project/API dependencies; not the full compiler/firmware closure.
    sources += [ROOT / path for path in (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_streaming_qktv.hpp",
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/q_chunk_remapping.hpp",
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp",
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp",
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp",
        "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp",
        "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp",
        "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp",
        "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl",
        "tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h",
        "tt_metal/hw/inc/api/compute/experimental/matmul_custom.h",
        "tt_metal/hw/inc/api/compute/experimental/sdpa_sub_custom.h",
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp",
        "tt_metal/hw/inc/api/compute/eltwise_unary/exp.h",
        "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h",
        "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp",
        "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl",
        "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp",
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce.cpp",
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp",
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp",
        "ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp",
        "ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_h_program_factory.cpp",
        "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py",
        "experiments/sdpa-l2/bfp4-lofi-v1/probe.py",
        "experiments/sdpa-l2/bfp4-lofi-v1/numerics.py",
        "experiments/sdpa-l2/frontier-accuracy-v1/run.py",
    )]
    return sorted(set(sources))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--destination", choices=("main_bf16", "fast_bf16"), default="main_bf16")
    parser.add_argument("--denom-only", action="store_true", help="FAST: compensate denominator only, not output")
    parser.add_argument("--kv-formats", choices=("b8_b4", "b4_b4"), default="b8_b4")
    parser.add_argument("--center-mode", choices=("none", "original_mean", "matched_mean"), default="none")
    parser.add_argument("--mean-mode", choices=("bf16_fpu", "fp32_sfpu"), default="bf16_fpu")
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--cores", type=int, default=110)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--distribution", default="normal")
    parser.add_argument("--common-mode", type=float, default=32)
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
    inputs = REPRO.make_inputs(args.heads, args.length, args.length, 128, args.seed, args.distribution, args.common_mode)
    rows = torch.linspace(0, args.length - 1, min(args.sample_rows, args.length)).long().unique()
    reference = REPRO.reference(inputs[0][..., rows, :], inputs[1], inputs[2])
    sources = source_files(args.destination)
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        originals, tensors, out, attention, preprocess, combined, info = build(device, args, inputs)
        attention.with_epilogue()
        actual = ttnn.to_torch(out)
        assert torch.isfinite(actual).all()
        accuracy = REPRO.metrics(actual[..., rows, :], reference)
        centered_accuracy = centered_output_metrics(actual[..., rows, :], reference, inputs[2])
        print("ACCURACY", json.dumps(accuracy), flush=True)
        print("CENTERED_OUTPUT_ACCURACY", json.dumps(centered_accuracy), flush=True)
        if args.max_l2 is not None:
            assert accuracy["l2_pct"] < args.max_l2, "Accuracy gate failed; do not time this candidate"
        attention_time = timed(device, attention, args)
        epilogue_time = timed(device, attention.epilogue, args) if args.center_mode != "none" else dict(median_ms=0.0, replay_ms=[])
        attention_epilogue_time = timed(device, attention.with_epilogue, args)
        preprocessing_time = timed(device, preprocess, args)
        combined_time = timed(device, combined, args)
        assert torch.equal(actual, ttnn.to_torch(out)), "Trace replay changed output"
        flops = 4 * args.heads * args.length**2 * 128
        attention_ms, combined_ms = attention_time["median_ms"], combined_time["median_ms"]
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == source_hashes[str(p.relative_to(ROOT))] for p in sources)
        epilogue_check = None
        if attention.epilogue_bias is not None:
            core = ttnn.to_torch(attention.core_output).float()
            bias = ttnn.to_torch(attention.epilogue_bias).float()
            expected_final = (core + bias).bfloat16()
            epilogue_check = dict(mismatch=int((actual != expected_final).sum()),
                oracle="BF16 RNE of actual BF16 core output plus actual BF16 compact bias")
            assert epilogue_check["mismatch"] == 0, epilogue_check
        record = dict(
            **vars(args), **info, accuracy=accuracy, centered_output_accuracy=centered_accuracy,
            sampled_query_rows=rows.tolist(),
            accuracy_scope="Original BF16 Q/K/V FP64 reference; all heads and KV, explicit sampled Q rows; all output finite",
            useful_flops=flops, attention=attention_time, preprocessing=preprocessing_time, combined=combined_time,
            epilogue=epilogue_time, attention_with_epilogue=attention_epilogue_time, epilogue_check=epilogue_check,
            attention_tflops=flops / (attention_ms * 1e9) if attention_ms else None,
            combined_tflops=flops / (combined_ms * 1e9) if combined_ms else None,
            trace_equal=True if args.iters else None,
            output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
            source_sha256=source_hashes,
            warning="Private chain reader; combined time includes device V mean/centering/quantization, matched-mode V4 decode and second mean, and BF16 output-bias epilogue; quantization is not free",
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("RESULT", json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
