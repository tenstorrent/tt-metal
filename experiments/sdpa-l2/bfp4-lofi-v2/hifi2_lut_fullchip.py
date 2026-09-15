# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated HiFi2/native-grid LUT-exp/BFP8 control.

Tests a finer K/V operand format with the same native-exp FP32 streaming path.
Fixed kernels, buffers and math mapping; not production dispatch. Original BF16
FP64 sampled-Q reference includes device input-preprocessing error and cost.
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


REPRO = module("fullchip_repro", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")
PREP = module("fullchip_preprocess", HERE / "preprocess.py")
B4_PREP = module("fullchip_b4_preprocess", HERE / "bfp4_round.py")
B8_PREP = module("fullchip_b8_preprocess", HERE / "bfp8_round.py")
Q_SCALE = module("fullchip_q_scale", HERE / "q_prescale.py")
CENTER = module("fullchip_center", HERE / "center_preprocess.py")
MEAN = module("fullchip_mean", HERE / "center_mean.py")


def source_hashes():
    import native_exp_qualification as qualification
    hashes = qualification.source_hashes()
    sources = [Path(__file__).resolve(), HERE / "bfp8_round.py",
               *sorted((HERE / "bfp8_round").glob("*.cpp")),
               ROOT / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp",
               ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h"]
    sources.extend([HERE / "exp_lut.hpp", HERE / "exp_lut_macro.hpp",
                    HERE / "exp_lut_macro_streaming/compute.cpp"])
    # Shared attention/reduction headers are explicitly inherited from qualification.
    hashes.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    return hashes


def build(device, args, inputs):
    lofi = args.variant.startswith("lofi_")
    hi2_b8 = args.variant == "hi2_fp32_b8"
    fp32 = args.variant in ("balanced", "accurate", "lofi_fp32", "lofi_fp32_b8", "hi2_fp32_b8", "lofi_fp32_b4")
    fast = args.variant in ("fast", "lofi_fast", "lofi_fast_b8", "lofi_fast_b4")
    compressed = args.variant.endswith("_b8")
    bfp4 = args.variant.endswith("_b4")
    qt, chunks = args.q_chunk // 32, args.length // 512
    jobs = args.heads * args.length // args.q_chunk
    hardware_grid = device.compute_with_storage_grid_size()
    cores = min(args.cores, jobs, hardware_grid.x * hardware_grid.y)
    coords = [ttnn.CoreCoord(i % hardware_grid.x, i // hardware_grid.x) for i in range(cores)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    originals = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in inputs]
    tensors, preprocessing = [], []
    assert args.q_prescale > 0
    assert args.q_prescale == 1.0 or lofi
    assert not args.center_k or (lofi and (compressed or bfp4))
    assert not args.b8_rne or hi2_b8
    mean_precision = None
    for i, src in enumerate(originals):
        if lofi or hi2_b8:
            bits = 7 if i == 0 else (8 if hi2_b8 else 5)
            bias = None
            if args.center_k and i == 1:
                bias, mean_invoke = MEAN.build(device, src, args.mean_mode)
                mean_invoke()
                preprocessing.append(mean_invoke)
                mean_precision = mean_invoke.precision
                tensor, invoke, _ = CENTER.build(device, src, bias, args.cores,
                                                  "b4_rne" if bfp4 else "b8_rne5")
            elif i == 0 and args.q_prescale != 1.0:
                tensor, invoke, _ = Q_SCALE.build(device, src, 7, "bf16", args.cores,
                                                   scale=args.q_prescale)
            elif args.b8_rne and i:
                tensor, invoke, _ = B8_PREP.build(device, src, ncores=args.cores)
            elif bfp4 and i:
                tensor, invoke, _ = B4_PREP.build(device, src, ncores=args.cores)
            else:
                tensor, invoke, _ = PREP.build(device, src, bits,
                                               "b8" if compressed and i else "bf16", args.cores,
                                               bfp8_pack_precise=args.bfp8_pack_precise)
            invoke()
            if args.check_preprocess:
                expected = PREP.MODEL.round_significand(inputs[i], bits)
                if compressed and i:
                    if not args.bfp8_pack_precise:
                        expected = PREP.MODEL.round_significand(expected, 7, "rna")
                    expected = PREP.MODEL.quantize(expected, 7, "device")
                if bfp4 and i:
                    expected = B4_PREP.host_rne_bfp4(inputs[i])
                if args.b8_rne and i:
                    expected = B8_PREP.host_rne_bfp8(inputs[i])
                if i == 0 and args.q_prescale != 1.0:
                    expected = PREP.MODEL.round_significand(inputs[i].float() * args.q_prescale, 7)
                if bias is not None:
                    expected = CENTER.quantize_oracle(CENTER.center_oracle(inputs[i], ttnn.to_torch(bias)),
                                                       "b4_rne" if bfp4 else "b8_rne5")
                actual_input = ttnn.to_torch(tensor).float()
                mismatch = int((actual_input != expected).sum())
                print("PREPROCESS_CHECK", i, mismatch, flush=True)
                assert mismatch == 0
            tensors.append(tensor)
            preprocessing.append(invoke)
        else:
            tensors.append(src)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, args.heads, args.length, 128]), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    slots = 1 if fp32 else 2
    state_format, state_bytes = (ttnn.float32, 4096) if fp32 else (ttnn.bfloat16, 2048)
    kv_format, kv_bytes = (ttnn.bfloat8_b, 1088) if compressed else (ttnn.bfloat16, 2048)
    if bfp4:
        kv_format, kv_bytes = ttnn.bfloat4_b, 576
    specs = [(0, 2 * qt * 4, 2048, ttnn.bfloat16), (1, 64 * slots, kv_bytes, kv_format),
             (2, 64 * slots, kv_bytes, kv_format), (3, 1, 2048, ttnn.bfloat16),
             (4, 1, 2048, ttnn.bfloat16), (5, 1, state_bytes, state_format),
             (6, qt * 16, state_bytes, state_format),
             (8, qt * 4 * (2 if fast else 1), state_bytes, state_format),
             (9, qt * 4 * (2 if fast else 1), state_bytes, state_format),
             (10, qt, 2048, ttnn.bfloat16), (11, qt, 2048, ttnn.bfloat16),
             (12, qt * (2 if fast else 1), state_bytes, state_format),
             (13, qt * (2 if fast else 1), state_bytes, state_format),
             (14, qt, state_bytes, state_format), (16, 8 if fp32 else 16, 2048, ttnn.bfloat16)]
    cbs = []
    for index, count, size, fmt in specs:
        formats = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=size)]
        if index == 6 and fp32:
            formats.append(ttnn.CBFormatDescriptor(buffer_index=7, data_format=fmt, page_size=size))
        cbs.append(ttnn.CBDescriptor(total_size=count * size, core_ranges=grid, format_descriptors=formats))
    defines = dict(EXP_APPROX_MODE="1", STATS_GRANULARITY="4" if fp32 else str(qt),
                   SUB_EXP_GRANULARITY="4" if fp32 else "8", MUL_BCAST_GRANULARITY="4" if fp32 else "8",
                   DHT_GRANULARITY="4", REDUCE_GRANULARITY="2" if fp32 else "4")
    if args.variant in ("main", "lofi_main", "lofi_main_b4"):
        defines["RESIDENT_MAIN"] = "1"
    if fast:
        defines.update(SDPA_STREAMING_ACCURACY="1", SDPA_STREAMING_NUMERATOR_COMPENSATION="1",
                       SDPA_OUT_A_CB="8", SDPA_OUT_B_CB="9")
        if args.fix_correction:
            defines["SDPA_LOFI_FIX_CORRECTION"] = "1"
    if fp32:
        defines.update(SDPA_FP32_STREAMING="1", SDPA_FP32_STATE="1", SDPA_HIFI2_ROUND="1")
    if args.variant == "balanced":
        defines["SDPA_QK4"] = "1"
    if args.variant == "accurate":
        defines.update({key: "1" for key in ("SDPA_FP32_FUSED_EXP", "SDPA_FP32_REUSE_EXP",
            "SDPA_FP32_EXTRA_CONST", "SDPA_FP32_PAIRED_UNPACK", "SDPA_FP32_PAIRED_PACK",
            "SDPA_FP32_L1_SUB", "SDPA_FP32_L1_MACRO", "SDPA_FP32_REFINE_MACRO")})
        defines.update(SDPA_DIAG_EXP_MODE="4", SDPA_FP32_SUB_BATCH="2", SDPA_DENOM_PHASES="2", SDPA_DIAG_SCORE_CB="7")
    if lofi:
        if fp32:
            defines["SDPA_MATCH_HIFI2"] = "1"
        else:
            defines["SDPA_LOFI_SAFE_RESCALE"] = "1"
    fidelity = ttnn.MathFidelity.LoFi if lofi else (ttnn.MathFidelity.HiFi4 if args.variant == "accurate" else ttnn.MathFidelity.HiFi2)
    if args.exp_degree != 3:
        assert lofi and fp32, "Exp refiner is FP32-only; BF16 calls native exp"
        defines["SDPA_LOFI_EXP_DEGREE"] = str(args.exp_degree)
    if args.native_exp:
        assert (lofi or hi2_b8) and fp32 and args.exp_degree == 3
        defines["SDPA_LOFI_NATIVE_EXP"] = "1"
        if args.lut_exp:
            defines.update(SDPA_LOFI_LUT_EXP="1", SDPA_LOFI_LUT_MACRO="1")
    config = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32,
                                         dst_full_sync_en=False, math_approx_mode=True)
    if fp32:
        pd = ttnn._ttnn.program_descriptor
        modes = pd.VectorUnpackToDestMode([pd.UnpackToDestMode.Default] * 64)
        for cb in (5, 7, 8, 9, 12, 13, 14):
            modes[cb] = pd.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
    read_args, write_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    semaphores = []
    if args.reader_chain:
        assert args.q_chunk == 256 and cores % args.heads == 0 and cores >= args.heads
        chain_length = cores // args.heads
        jobs_per_head = args.length // args.q_chunk
        assert chain_length <= jobs_per_head
        physical = [device.worker_core_from_logical_core(c) for c in coords]
        semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=value)
                      for i, value in enumerate((0, 0, 1))]
    offset, counts = 0, []
    for i, c in enumerate(coords):
        count = jobs // cores + (i < jobs % cores)
        chain_args = []
        if args.reader_chain:
            head, rank = divmod(i, chain_length)
            count = jobs_per_head // chain_length + (rank < jobs_per_head % chain_length)
            offset = head * jobs_per_head + rank * (jobs_per_head // chain_length) + min(rank, jobs_per_head % chain_length)
            prev = physical[i - 1] if rank else ttnn.CoreCoord(0, 0)
            following = physical[i + 1] if rank + 1 < chain_length else ttnn.CoreCoord(0, 0)
            next_count = (jobs_per_head // chain_length + (rank + 1 < jobs_per_head % chain_length)) if rank + 1 < chain_length else 0
            chain_args = [rank, chain_length, prev.x, prev.y, following.x, following.y, next_count]
        read_args[c.x][c.y] = [t.buffer_address() for t in tensors] + [offset, count] + chain_args
        write_args[c.x][c.y] = [out.buffer_address(), offset, count]
        compute_args[c.x][c.y] = [count]
        offset += count
        counts.append(count)
    reader_cta = [qt, chunks, args.length // args.q_chunk]
    reader_defines = {"SDPA_READER_BARRIER_TILES": str(args.read_barrier_tiles)}
    if args.reader_split:
        reader_defines["SDPA_READER_SPLIT_KV"] = "1"
    if args.reader_linear_k:
        reader_defines["SDPA_READER_LINEAR_K"] = "1"
    for tensor in tensors:
        reader_cta += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    reader_source = "reader_chain.cpp" if args.reader_chain else "reader.cpp"
    desc = ttnn.ProgramDescriptor(cbs=cbs, semaphores=semaphores, kernels=[
        ttnn.KernelDescriptor(kernel_source=PREFIX + reader_source, core_ranges=grid,
                              compile_time_args=reader_cta, runtime_args=read_args,
                              defines=list(reader_defines.items()), config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=PREFIX + "writer.cpp", core_ranges=grid,
                              compile_time_args=[qt] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                              runtime_args=write_args, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source="experiments/sdpa-l2/bfp4-lofi-v2/exp_lut_macro_streaming/compute.cpp", core_ranges=grid,
                              compile_time_args=[chunks, struct.unpack("I", struct.pack("f", 1 / math.sqrt(128) / args.q_prescale))[0], qt],
                              runtime_args=compute_args, defines=list(defines.items()), config=config)])
    def attention():
        ttnn.generic_op(tensors + [out], desc)
    def preprocess():
        for invoke in preprocessing:
            invoke()
    def combined():
        preprocess()
        attention()
    info = dict(actual_cores=cores, q_jobs=jobs, jobs_per_core=counts, input_slots=slots,
                cb_bytes_per_core=sum(n * size for _, n, size, _ in specs), defines=defines,
                fidelity=str(fidelity), fp32_dst=fp32, device_preprocessing=lofi or hi2_b8,
                device_k_mean_precision=mean_precision,
                input_storage="BF16 Q; BFP4 K/V" if bfp4 else ("BF16 Q; BFP8 K/V" if compressed else "BF16 Q/K/V"))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--lut-exp", action="store_true")
    parser.add_argument("--variant", choices=("main", "fast", "balanced", "accurate", "lofi_main",
                                             "lofi_fast", "lofi_fast_b8", "lofi_fp32", "lofi_fp32_b8", "hi2_fp32_b8",
                                             "lofi_fast_b4", "lofi_main_b4", "lofi_fp32_b4"), required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--q-chunk", type=int, choices=(128, 256), default=256)
    parser.add_argument("--cores", type=int, default=110)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--distribution", default="normal")
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--check-preprocess", action="store_true")
    parser.add_argument("--q-prescale", type=float, default=1.0)
    parser.add_argument("--b8-rne", action="store_true", help="HiFi2 BFP8 control: RNE shared-exponent K/V before packing")
    parser.add_argument("--bfp8-pack-precise", action="store_true", help="Use supported precise BFP8 pack source in input preprocessing")
    parser.add_argument("--exp-degree", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--native-exp", action="store_true")
    parser.add_argument("--center-k", action="store_true", help="Include device mean and fused center+quantize")
    parser.add_argument("--mean-mode", choices=("bf16_fpu", "fp32_sfpu"), default="bf16_fpu")
    parser.add_argument("--fix-correction", action="store_true")
    parser.add_argument("--reader-split", action="store_true", help="Publish K independently before reserving V")
    parser.add_argument("--reader-chain", action="store_true", help="One unicast K/V forwarding chain per head; fixed Q256")
    parser.add_argument("--reader-linear-k", action="store_true", help="Read K in linear DRAM order, scatter transposed L1 grid")
    parser.add_argument("--read-barrier-tiles", type=int, choices=(0, 1, 2, 4, 8, 16, 32, 64), default=0)
    parser.add_argument("--max-l2", type=float)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--trace-repeats", type=int, default=1)
    args = parser.parse_args()
    assert args.variant == "hi2_fp32_b8", "This isolated driver is HiFi2/BFP8 only"
    assert args.native_exp, "Use the original fullchip driver for polynomial exp controls"
    assert args.length % 512 == 0 and args.iters >= 0
    assert args.q_chunk == 256 and args.q_prescale == 1.0 and not args.center_k
    assert Path(args.label).name == args.label
    path = HERE / (args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    pins = source_hashes()
    torch.set_num_threads(8)
    inputs = REPRO.make_inputs(args.heads, args.length, args.length, 128, args.seed, args.distribution)
    rows = torch.linspace(0, args.length - 1, min(args.sample_rows, args.length)).long().unique()
    reference = REPRO.reference(inputs[0][..., rows, :], inputs[1], inputs[2])
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
        trace = ttnn.begin_trace_capture(device, cq_id=0)
        combined()
        ttnn.end_trace_capture(device, trace, cq_id=0)
        try:
            for _ in range(2):
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                assert torch.equal(actual, ttnn.to_torch(out)), "Trace replay changed output"
        finally:
            ttnn.release_trace(device, trace)
        attention_time = timed(device, attention, args)
        preprocessing_time = timed(device, preprocess, args) if info["device_preprocessing"] else None
        combined_time = timed(device, combined, args) if info["device_preprocessing"] else attention_time
        assert torch.equal(actual, ttnn.to_torch(out)), "Trace replay changed output"
        flops = 4 * args.heads * args.length**2 * 128
        assert pins == source_hashes(), "Sources changed during this run"
        record = dict(**vars(args), **info, accuracy=accuracy, sampled_query_rows=rows.tolist(),
                      accuracy_scope="All heads and KV; only explicit sampled Q rows; all output values checked finite",
                      useful_flops=flops, attention=attention_time, preprocessing=preprocessing_time,
                      combined=combined_time, attention_tflops=flops / (attention_time["median_ms"] * 1e9) if args.iters else None,
                      combined_tflops=flops / (combined_time["median_ms"] * 1e9) if args.iters else None,
                      trace_equal=True, output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
                      source_sha256=pins, sources_unchanged=True, finite=True,
                      warning="Experimental common reader, not production SDPA performance; noncausal square attention")
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("RESULT", json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)
