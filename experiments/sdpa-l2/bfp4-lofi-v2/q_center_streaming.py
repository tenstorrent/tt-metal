# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated Q128/H1 LoFi attention with high-precision mean-Q correction.

Uses half-sync FP32 DST with a separate two-score/two-correction SFPU pass.
Producer and Q/K/V preprocessing costs are measured separately and together.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import struct
from pathlib import Path

import torch

import bfp4_round
import center_mean
import preprocess
import q_center_preprocess
import q_center_producer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/q_center_streaming/"


def build(device, inputs, kv_format="b8", ncores=4, center=True, mean_mode="bf16_fpu", skip_add=False, half_sync=True, logical_dst4=False):
    """Return out, attention, preprocess_all, combined, info.

    Original BF16 inputs [Q128,K_N,V_N], H1,D128. preprocess_all includes
    device mean, centering, correction and K/V quantization; no free producer.
    All device tensors and producer closures stay live through these callables.
    """
    import ttnn

    assert kv_format in ("b8", "b4")
    assert tuple(inputs[0].shape) == (1, 1, 128, 128)
    assert tuple(inputs[1].shape) == tuple(inputs[2].shape)
    length = inputs[1].shape[2]
    assert tuple(inputs[1].shape) == (1, 1, length, 128) and length > 0 and length % 512 == 0
    assert all(x.dtype == torch.bfloat16 for x in inputs)
    original = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in inputs]
    if center:
        q, correction, producer, producer_info = q_center_producer.build(device, original[0], original[1], ncores, mean_mode)
    else:
        q, producer, _ = preprocess.build(device, original[0], 7, "bf16", ncores)
        correction = ttnn.zeros([1, 1, 32, length], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        producer_info = dict(control="Q RNE7 without centering; device-zero correction still injected")
    kv, prepares = [], []
    for src in original[1:]:
        if kv_format == "b8":
            tensor, call, _ = preprocess.build(device, src, 5, "b8", ncores)
        else:
            tensor, call, _ = bfp4_round.build(device, src, ncores=ncores)
        kv.append(tensor)
        prepares.append(call)
    out = ttnn.allocate_tensor_on_device([1, 1, 128, 128], ttnn.bfloat16,
                                        ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    core = ttnn.CoreCoord(0, 0)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    kv_dtype, kv_bytes = (ttnn.bfloat8_b, 1088) if kv_format == "b8" else (ttnn.bfloat4_b, 576)
    specs = [(0, 32, 2048, ttnn.bfloat16),
             (1, 128, kv_bytes, kv_dtype), (2, 128, kv_bytes, kv_dtype),
             (3, 1, 2048, ttnn.bfloat16), (4, 1, 2048, ttnn.bfloat16),
             (5, 1, 4096, ttnn.float32), (6, 64, 4096, ttnn.float32),
             (8, 16, 4096, ttnn.float32), (9, 16, 4096, ttnn.float32),
             (10, 4, 2048, ttnn.bfloat16), (11, 4, 2048, ttnn.bfloat16),
             (12, 4, 4096, ttnn.float32), (13, 4, 4096, ttnn.float32),
             (14, 4, 4096, ttnn.float32), (16, 8, 2048, ttnn.bfloat16),
             (20, 32, 4096, ttnn.float32)]
    cbs = []
    for index, pages, size, dtype in specs:
        formats = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)]
        if index == 6:
            formats.append(ttnn.CBFormatDescriptor(buffer_index=7, data_format=dtype, page_size=size))
        cbs.append(ttnn.CBDescriptor(total_size=pages * size, core_ranges=grid, format_descriptors=formats))
    defines = dict(EXP_APPROX_MODE="1", STATS_GRANULARITY="4", SUB_EXP_GRANULARITY="4",
        MUL_BCAST_GRANULARITY="4", DHT_GRANULARITY="4", REDUCE_GRANULARITY="2",
        SDPA_FP32_STREAMING="1", SDPA_FP32_STATE="1", SDPA_HIFI2_ROUND="1",
        SDPA_LOFI_DENOM="1", SDPA_DIAG_EXP_MODE="1")
    if skip_add:
        assert not center, "Skip-add is only a no-centering diagnostic"
        defines["SDPA_Q_CENTER_SKIP_ADD"] = "1"
    assert half_sync or skip_add, "Correction requires half-sync; full-sync is a known-failing skip-add diagnostic only"
    if logical_dst4:
        defines["SDPA_Q_CENTER_LOGICAL_DST4"] = "1"
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=True, dst_full_sync_en=not half_sync, math_approx_mode=True)
    pd = ttnn._ttnn.program_descriptor
    modes = pd.VectorUnpackToDestMode([pd.UnpackToDestMode.Default] * 64)
    for cb in (5, 7, 8, 9, 12, 13, 14, 20):
        modes[cb] = pd.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    tensors = [q, *kv, correction, out]
    reader, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader[0][0] = [x.buffer_address() for x in tensors[:4]]
    writer[0][0] = [out.buffer_address()]
    cta = [length // 512]
    for tensor in tensors[:4]:
        cta += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    desc = ttnn.ProgramDescriptor(cbs=cbs, semaphores=[], kernels=[
        ttnn.KernelDescriptor(kernel_source=PREFIX + "reader.cpp", core_ranges=grid,
            compile_time_args=cta, runtime_args=reader, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=PREFIX + "writer.cpp", core_ranges=grid,
            compile_time_args=ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=writer, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=PREFIX + "compute.cpp", core_ranges=grid,
            compile_time_args=[length // 512, struct.unpack("I", struct.pack("f", 1 / math.sqrt(128)))[0]],
            defines=list(defines.items()), config=config)])

    def attention():
        ttnn.generic_op(tensors, desc)

    def preprocess_all():
        producer()
        for call in prepares:
            call()

    def combined():
        preprocess_all()
        attention()

    preprocess_all.producer = producer
    preprocess_all.prepared = tensors[:4]
    info = dict(producer=producer_info, fp32_dst=True, dst_full_sync_en=not half_sync,
        synchronization_warning="half-sync FP32, four slots; separate correction pass" if half_sync else "KNOWN-FAILING full-sync diagnostic; no accepted performance claim",
        q_chunk=128, k_chunk=512, input_slots=dict(q=2, k=2, v=2, correction=2),
        cb_bytes_per_core=sum(p * s for _, p, s, _ in specs), defines=defines,
        correction_add="bypassed diagnostic" if skip_add else "full FP32 SFPU;2score+2correction; after initial QK pack, before max/scale; QK only",
        correction_score_spill="FP32 CB6/CB7 alias, fixed64-page Q128 arena; no BF16 score rounding",
        score_and_p="FP32; unbiased cheap exp; LoFi PV and denominator consume matching SrcB bits",
        recurrent_state="FP32; BF16 max", correction_storage_bytes=32 * length * 4,
        attention_cores=1, producer_not_free=True)
    return out, attention, preprocess_all, combined, info


def source_files():
    candidate = ROOT / "experiments/sdpa-l2/hybrid-mixed-v1/candidate"
    files = q_center_producer.source_files() + [Path(__file__).resolve(), HERE / "preprocess.py",
        HERE / "numerics.py", HERE / "bfp4_round.py",
        candidate / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp",
        candidate / "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h"]
    for directory in ("q_center_streaming", "preprocess", "bfp4_round"):
        files += sorted((HERE / directory).glob("*.cpp")) + sorted((HERE / directory).glob("*.hpp"))
    return list(dict.fromkeys(files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--kv-format", choices=("b8", "b4"), default="b8")
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--skip-add", action="store_true", help="No-centering diagnostic: omit all correction copy/add/reinit work")
    sync = parser.add_mutually_exclusive_group()
    sync.add_argument("--half-sync", action="store_true", dest="half_sync", default=True,
                      help="Default: original half-sync geometry with correction postpass")
    sync.add_argument("--full-sync", action="store_false", dest="half_sync",
                      help="Known-failing skip-add diagnostic only; not valid performance evidence")
    parser.add_argument("--logical-dst4", action="store_true", help="Keep PV/normalization scheduling at four logical DST tiles")
    parser.add_argument("--distribution", choices=("normal", "common_q", "structured"), default="normal")
    parser.add_argument("--common-mode", type=float, default=32)
    parser.add_argument("--mean-mode", choices=("bf16_fpu", "fp32_sfpu"), default="bf16_fpu")
    parser.add_argument("--cores", type=int, default=4, help="Preprocessor cores; attention always one core")
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--max-l2", type=float, default=100)
    parser.add_argument("--max-correction-l2", type=float, default=0.01,
                        help="Producer gate; explicit override required while TTNN FP32 matmul floor is investigated")
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trace-repeats", type=int, default=10)
    args = parser.parse_args()
    assert args.length > 0 and args.length % 512 == 0 and args.cores > 0
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats >= 0
    assert Path(args.label).name == args.label
    path = HERE / (args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    spec = importlib.util.spec_from_file_location("q_center_stream_repro", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")
    repro = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repro)
    inputs = repro.make_inputs(1, 128, args.length, 128, args.seed,
        "common_q" if args.distribution == "common_q" else "normal", args.common_mode)
    if args.distribution == "structured":
        inputs[0] = (inputs[0].float() + torch.sin(torch.arange(128).float() / 9) * args.common_mode).bfloat16()
    reference = repro.reference(*inputs)
    files = source_files()
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    import ttnn

    device = ttnn.open_device(device_id=0, trace_region_size=8388608 if args.iters else 0)
    try:
        out, attention, prep, combined, info = build(device, inputs, args.kv_format, args.cores,
            not args.no_center, args.mean_mode, args.skip_add, args.half_sync, args.logical_dst4)
        prep()
        q, k, v, correction = [ttnn.to_torch(x).float() for x in prep.prepared]
        if not args.no_center:
            bias = ttnn.to_torch(prep.producer.bias).bfloat16()
            assert torch.equal(q, q_center_preprocess.oracle(inputs[0], bias))
            c_ref = bias.double() @ inputs[1].double().transpose(-1, -2)
            c_metrics = q_center_producer.metrics(correction, c_ref)
            assert c_metrics["finite"] and c_metrics["l2_pct"] < args.max_correction_l2
        else:
            assert torch.equal(q, preprocess.MODEL.round_significand(inputs[0], 7))
            assert bool((correction == 0).all())
            c_metrics = dict(control="device zeros")
        assert torch.equal(correction, correction[:, :, :1].expand_as(correction))
        attention()
        actual = ttnn.to_torch(out)
        assert bool(torch.isfinite(actual).all()), "Nonfinite attention output; do not time this candidate"
        accuracy = repro.metrics(actual, reference)
        print("Q_CENTER_STREAMING_ACCURACY", json.dumps(accuracy), flush=True)
        assert accuracy["l2_pct"] < args.max_l2
        timings = {}
        for name, call in (("preprocess_all", prep), ("attention_only", attention), ("combined", combined)):
            times = center_mean.measure(device, call, args)
            timings[name] = dict(ms=times, median_ms=statistics.median(times) if times else None)
        assert torch.equal(actual, ttnn.to_torch(out)), "Trace replay changed output"
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in files)
        result = dict(**vars(args), **info, accuracy=accuracy, correction_metrics=c_metrics, timings=timings,
            original_input_sha256=[hashlib.sha256(x.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest() for x in inputs],
            prepared_sha256=[hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() for x in (q, k, v, correction)],
            source_sha256=hashes, reference="all128 Q rows; FP64 attention on original BF16 Q/K/V", trace_equal=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print("Q_CENTER_STREAMING_RESULT", json.dumps(result), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
