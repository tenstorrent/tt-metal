# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated LoFi FP32 streaming prototype; Q128/K512/D128."""

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
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/streaming/"

v2_spec = importlib.util.spec_from_file_location("lofi_model", HERE / "numerics.py")
V2 = importlib.util.module_from_spec(v2_spec)
v2_spec.loader.exec_module(V2)


def run(device, args, inputs, ref):
    accurate = args.mode in (
        "accurate",
        "fp32_hifi2",
        "fp32_hifi2_cheap",
        "qk4_pv2",
        "qk4_pv2_fullsub",
        "qk2_pv2_fullsub",
    )
    fast = args.mode == "fast"
    hybrid = args.mode == "hybrid"
    q, k, v = inputs
    original_v = v
    if getattr(args, "center_k", False):
        k = k.double() - k.double().mean(-2, keepdim=True)
    if getattr(args, "center_v", False):
        assert args.v_mean_correction
        v = v.double() - v.double().mean(-2, keepdim=True)
    qfmt, kfmts, pfmt, vfmts = V2.VARIANTS[args.variant]
    q_prescale = getattr(args, "q_prescale", 1.0)
    assert q_prescale > 0 and (q_prescale == 1.0 or qfmt == "e7")
    residual = len(kfmts) == 2
    assert len(kfmts) == len(vfmts) and pfmt in ("fp32", "e7")
    assert not residual or accurate
    formats = (qfmt, kfmts[0], vfmts[0])
    prepared = (q, k, v)
    device_residual = residual and getattr(args, "device_preprocess", False)
    if residual:
        assert args.exp_quality == "accurate" or getattr(args, "no_p_round", False)
        if not device_residual:
            kc = V2.components(k, kfmts, args.qkv_route)
            vc = V2.components(v, vfmts, args.qkv_route)
            prepared = (q, kc[0], vc[0], kc[1], vc[1])
        formats += (kfmts[1], vfmts[1])
    tensors, input_specs = [], []
    preprocessing_inputs = []
    preprocessing_calls = []
    if device_residual:
        assert not getattr(args, "center_k", False) and not getattr(args, "center_v", False)
        assert kfmts == vfmts and kfmts in (("b4", "b4"), ("b4", "e5_b8"))
        modules = []
        for name in (("q_prescale" if q_prescale != 1.0 else "preprocess"), "bfp4_residual_preprocess"):
            spec = importlib.util.spec_from_file_location("fused_" + name, HERE / (name + ".py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            modules.append(mod)
        prep, residual_prep = modules
        preprocessing_inputs = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in (q, k, v)]
        tq, iq, _ = prep.build(
            device, preprocessing_inputs[0], 7, "bf16", 110, **({"scale": q_prescale} if q_prescale != 1.0 else {})
        )
        options = dict(components=2, ncores=110)
        if kfmts[1] == "e5_b8":
            options["second_format"] = "b8_rne5"
        tk, ik, _ = residual_prep.build(device, preprocessing_inputs[1], **options)
        tv, iv, _ = residual_prep.build(device, preprocessing_inputs[2], **options)
        preprocessing_calls = [iq, ik, iv]
        for call in preprocessing_calls:
            call()
        tensors = [tq, tk[0], tv[0], tk[1], tv[1]]
        input_specs = [
            (t.dtype, 2048 if t.dtype == ttnn.bfloat16 else (576 if t.dtype == ttnn.bfloat4_b else 1088))
            for t in tensors
        ]
    for x, fmt in ([] if device_residual else zip(prepared, formats)):
        device_preprocess = getattr(args, "device_preprocess", False)
        original = x
        if fmt in ("e7", "e5", "e5_b8") and not device_preprocess:
            if fmt == "e7":
                x = x.float() * q_prescale
            x = V2.round_significand(x, 7 if fmt == "e7" else 5).bfloat16()
        if fmt == "b4":
            dtype, tile_bytes = ttnn.bfloat4_b, 576
        elif fmt in ("b8", "e5_b8"):
            dtype, tile_bytes = ttnn.bfloat8_b, 1088
        else:
            dtype, tile_bytes = ttnn.bfloat16, 2048
        if device_preprocess:
            assert fmt in ("e7", "e5", "e5_b8")
            scale_q = fmt == "e7" and q_prescale != 1.0
            prep_spec = importlib.util.spec_from_file_location(
                "lofi_preprocess", HERE / ("q_prescale.py" if scale_q else "preprocess.py")
            )
            prep = importlib.util.module_from_spec(prep_spec)
            prep_spec.loader.exec_module(prep)
            src = ttnn.from_torch(original, device=device, layout=ttnn.TILE_LAYOUT)
            tensor, preprocess_call, _ = prep.build(
                device,
                src,
                7 if fmt == "e7" else 5,
                "b8" if fmt == "e5_b8" else "bf16",
                110,
                **({"scale": q_prescale} if scale_q else {}),
            )
            preprocessing_inputs.append(src)
            preprocessing_calls.append(preprocess_call)
            preprocess_call()
            tensors.append(tensor)
        else:
            tensors.append(ttnn.from_torch(x.float(), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT))
        input_specs.append((dtype, tile_bytes))
    if getattr(args, "v_mean_correction", False):
        assert accurate and not getattr(args, "device_preprocess", False)
        represented_v = V2.effective(ttnn.to_torch(tensors[2]), vfmts[0], "right")
        if residual:
            represented_v += V2.effective(ttnn.to_torch(tensors[4]), vfmts[1], "right")
        bias = original_v.double().mean(-2, keepdim=True) - represented_v.mean(-2, keepdim=True)
        bias = bias.float().expand(1, 1, 32, 128).contiguous()
        tensors.append(ttnn.from_torch(bias, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT))
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 128, 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    kv_slots = 1 if accurate else 2
    fp = ttnn.float32 if accurate else ttnn.bfloat16
    page = 4096 if accurate else 2048
    specs = [
        (0, 32, 2048, ttnn.bfloat16),
        (1, 64 * kv_slots, 2048, ttnn.bfloat16),
        (2, 64 * kv_slots, 2048, ttnn.bfloat16),
        (3, 1, 2048, ttnn.bfloat16),
        (4, 1, 2048, ttnn.bfloat16),
        (5, 1, page, fp),
        (6, 64, page, fp),
        (8, 16 * (2 if fast else 1), page, fp),
        (9, 16 * (2 if fast else 1), page, fp),
        (10, 4, 2048, ttnn.bfloat16),
        (11, 4, 2048, ttnn.bfloat16),
        (12, 4 * (2 if fast else 1), page, fp),
        (13, 4 * (2 if fast else 1), page, fp),
        (14, 4, page, fp),
        (16, 8 if accurate else 16, 2048, ttnn.bfloat16),
    ]
    specs = [
        (idx, n, input_specs[idx][1], input_specs[idx][0]) if idx < 3 else (idx, n, size, dtype)
        for idx, n, size, dtype in specs
    ]
    if residual:
        specs += [(17 + i, 64 * kv_slots, input_specs[3 + i][1], input_specs[3 + i][0]) for i in range(2)]
    if getattr(args, "v_mean_correction", False):
        specs.append((19, 4, 4096, ttnn.float32))
    cbs = []
    if hybrid:
        specs = [
            (idx, n, 4096, ttnn.float32) if idx in (5, 8, 9, 12, 13, 14) else (idx, n, size, dtype)
            for idx, n, size, dtype in specs
        ]
    for idx, n, tile_bytes, dtype in specs:
        formats = [ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=tile_bytes)]
        if idx == 6 and accurate:
            formats.append(ttnn.CBFormatDescriptor(buffer_index=7, data_format=ttnn.float32, page_size=tile_bytes))
        if hybrid and idx in (12, 13):
            formats.append(ttnn.CBFormatDescriptor(buffer_index=idx + 5, data_format=ttnn.float32, page_size=4096))
        cbs.append(ttnn.CBDescriptor(total_size=n * tile_bytes, core_ranges=grid, format_descriptors=formats))
    defines = {
        "EXP_APPROX_MODE": "1",
        "STATS_GRANULARITY": "4",
        "SUB_EXP_GRANULARITY": "4" if accurate else "8",
        "MUL_BCAST_GRANULARITY": "4" if accurate else "8",
        "DHT_GRANULARITY": "4",
        "REDUCE_GRANULARITY": "2" if accurate else "4",
    }
    if args.mode == "main":
        defines["RESIDENT_MAIN"] = "1"
    if fast:
        defines.update(
            SDPA_STREAMING_ACCURACY="1", SDPA_STREAMING_NUMERATOR_COMPENSATION="1", SDPA_OUT_A_CB="8", SDPA_OUT_B_CB="9"
        )
    if accurate:
        defines.update(
            {
                name: "1"
                for name in (
                    "SDPA_FP32_STREAMING",
                    "SDPA_FP32_STATE",
                    "SDPA_HIFI2_ROUND",
                    "SDPA_FP32_FUSED_EXP",
                    "SDPA_FP32_REUSE_EXP",
                    "SDPA_FP32_EXTRA_CONST",
                    "SDPA_FP32_PAIRED_UNPACK",
                    "SDPA_FP32_PAIRED_PACK",
                    "SDPA_FP32_L1_SUB",
                    "SDPA_FP32_L1_MACRO",
                    "SDPA_FP32_REFINE_MACRO",
                )
            }
        )
        defines.update(SDPA_DIAG_EXP_MODE="4", SDPA_FP32_SUB_BATCH="2", SDPA_DENOM_PHASES="2", SDPA_DIAG_SCORE_CB="7")
    if args.mode in ("fp32_hifi2_cheap", "qk4_pv2"):
        defines = {k: v for k, v in defines.items() if not k.startswith("SDPA_")}
        defines.update(SDPA_FP32_STREAMING="1", SDPA_FP32_STATE="1", SDPA_HIFI2_ROUND="1")
    if args.mode.startswith("qk4_"):
        defines["SDPA_QK4"] = "1"
    if args.mode.endswith("_fullsub"):
        defines["SDPA_MATCH_HIFI2"] = "1"
        defines.pop("SDPA_DENOM_PHASES", None)
    if hybrid:
        defines["SDPA_HYBRID_STATE"] = "1"
        if args.hybrid_block_pack:
            defines["SDPA_HYBRID_BLOCK_PACK"] = "1"
    defines["SDPA_MATCH_HIFI2"] = "1"  # LoFi denominator matches seven-bit SrcB P.
    if input_specs[0][0] != ttnn.bfloat16:
        defines["SDPA_LOFI_Q_RECONFIG"] = "1"
    defines.pop("SDPA_DENOM_PHASES", None)
    if residual:
        defines.pop("SDPA_MATCH_HIFI2", None)
        defines.update(SDPA_LOFI_RESIDUALS="1", SDPA_LOFI_DENOM="1")
        if getattr(args, "skip_residual_reconfig", False):
            assert kfmts[0] == kfmts[1] and vfmts[0] == vfmts[1]
            defines["SDPA_LOFI_SAME_FORMAT_RESIDUAL"] = "1"
    if pfmt == "e7" or getattr(args, "round_p", False):
        assert args.exp_quality == "accurate" or getattr(args, "no_p_round", False)
        defines.pop("SDPA_MATCH_HIFI2", None)
        defines["SDPA_LOFI_DENOM"] = "1"
        if not getattr(args, "no_p_round", False):
            defines["SDPA_LOFI_ROUND_P"] = "1"
    if getattr(args, "safe_rescale", False):
        defines["SDPA_LOFI_SAFE_RESCALE"] = "1"
    if getattr(args, "v_mean_correction", False):
        defines["SDPA_LOFI_V_BIAS"] = "1"
    if args.exp_degree != 3:
        assert (
            args.destination == "fp32" and args.exp_quality == "cheap"
        ), "Exp refiner is FP32-only; BF16 calls native exp"
        defines["SDPA_LOFI_EXP_DEGREE"] = str(args.exp_degree)
    if args.native_exp:
        assert args.destination == "fp32" and args.exp_quality == "cheap" and args.exp_degree == 3
        defines["SDPA_LOFI_NATIVE_EXP"] = "1"
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, args.fidelity),
        fp32_dest_acc_en=accurate,
        dst_full_sync_en=False,
        math_approx_mode=True,
    )
    if accurate or hybrid:
        pd = ttnn._ttnn.program_descriptor
        modes = pd.VectorUnpackToDestMode([pd.UnpackToDestMode.Default] * 64)
        for cb in ((5, 8, 9, 12, 13, 14) if hybrid else (5, 7, 8, 9, 12, 13, 14)):
            modes[cb] = pd.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
        if getattr(args, "v_mean_correction", False):
            modes[19] = pd.UnpackToDestMode.UnpackToDestFp32
            config.unpack_to_dest_mode = modes
    reader, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader[0][0] = [t.buffer_address() for t in tensors]
    writer[0][0] = [out.buffer_address()]
    reader_cta = [args.q_repeats, args.k_chunks, kv_slots]
    for t in tensors:
        reader_cta += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    desc = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=[],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "reader.cpp",
                core_ranges=grid,
                compile_time_args=reader_cta,
                runtime_args=reader,
                defines=[(k, v) for k, v in defines.items() if k in ("SDPA_LOFI_RESIDUALS", "SDPA_LOFI_V_BIAS")],
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "writer.cpp",
                core_ranges=grid,
                compile_time_args=[args.q_repeats] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=writer,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "compute.cpp",
                core_ranges=grid,
                compile_time_args=[
                    args.q_repeats,
                    args.k_chunks,
                    struct.unpack("I", struct.pack("f", 1 / math.sqrt(128) / q_prescale))[0],
                ],
                defines=list(defines.items()),
                config=config,
            ),
        ],
    )
    sources = (
        list((ROOT / PREFIX).glob("*.cpp"))
        + list((ROOT / PREFIX).glob("**/*.hpp"))
        + list((ROOT / PREFIX).glob("**/*.h"))
        + [
            Path(__file__).resolve(),
            HERE / "safe_rescale.hpp",
            HERE / "round_p.hpp",
            HERE / "v_bias.hpp",
            HERE / "exp_refiner.hpp",
            HERE / "exp_native.hpp",
            HERE / "numerics.py",
        ]
    )
    if getattr(args, "device_preprocess", False):
        sources += [HERE / "preprocess.py", *sorted((HERE / "preprocess").glob("*.cpp"))]
        if q_prescale != 1.0:
            sources += [HERE / "q_prescale.py", *sorted((HERE / "q_prescale").glob("*.cpp"))]
    if device_residual:
        sources += [
            HERE / "bfp4_residual_preprocess.py",
            *sorted((HERE / "bfp4_residual_preprocess").glob("*.cpp")),
            *sorted((HERE / "bfp4_residual_preprocess").glob("*.hpp")),
        ]
    provenance = dict(
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        defines=defines,
        cb_specs=[(idx, n, size, str(dtype)) for idx, n, size, dtype in specs],
        arguments=vars(args),
    )
    (HERE / (args.label + ".provenance.json")).write_text(json.dumps(provenance, indent=2) + "\n")

    def invoke():
        ttnn.generic_op(tensors + [out], desc)

    invoke()
    actual = ttnn.to_torch(out)
    torch.save(actual, HERE / (args.label + ".output.pt"))
    print("OUTPUT_FINITE", int(torch.isfinite(actual).sum()), actual.numel(), flush=True)
    assert torch.isfinite(actual).all()
    l2 = 100 * ((actual.double() - ref).norm() / ref.norm()).item()

    times = []
    if args.iters:
        trace = ttnn.begin_trace_capture(device, cq_id=0)
        invoke()
        ttnn.end_trace_capture(device, trace, cq_id=0)
        try:
            for i in range(args.warmup + args.iters):
                start = time.perf_counter()
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                if i >= args.warmup:
                    times.append(1000 * (time.perf_counter() - start))
            assert torch.equal(actual, ttnn.to_torch(out))
        finally:
            ttnn.release_trace(device, trace)
    flops = 4 * 128 * 512 * 128 * args.q_repeats * args.k_chunks
    median = statistics.median(times) if times else None
    record = dict(
        **vars(args),
        cores=1,
        q_chunk=128,
        k_chunk=512,
        dim=128,
        kv_slots=kv_slots,
        cb_bytes=sum(n * size for _, n, size, _ in specs),
        useful_flops=flops,
        median_ms=median,
        trace_replay_ms=times,
        tflops_per_core=flops / (median * 1e9) if median else None,
        l2_pct=l2,
        pcc=None,
        full_output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
        trace_equal=bool(times),
        warning=(
            "Distinct K/V correctness test, not a no-DM throughput measurement"
            if args.distinct_kv
            else "Repeated resident Q/K/V, not general-input accuracy qualification or measured chip throughput"
        ),
    )

    record.update(repro.metrics(actual, ref))
    record["bf16_output_floor_l2_pct"] = repro.metrics(ref.bfloat16(), ref)["l2_pct"]
    return record, actual


spec = importlib.util.spec_from_file_location(
    "sdpa_repro", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py"
)
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "pervalue_rawp",
            "pervalue_rne",
            "kv4_rawp",
            "b8_rne5_rawp",
            "q8_only_rawp",
            "k8_only_rawp",
            "v8_only_rawp",
            "q7kv8_rawp",
            "q7kv8_full",
            "bf16kv8_full",
            "residual48_pervalue",
            "residual44_pervalue",
        ),
        default="pervalue_rawp",
    )
    parser.add_argument("--fidelity", choices=("LoFi", "HiFi2"), default="LoFi")
    parser.add_argument("--q-prescale", type=float, default=1.0)
    parser.add_argument("--no-p-round", action="store_true")
    parser.add_argument(
        "--round-p", action="store_true", help="Explicit RNE7 P with unbiased exp and matched denominator"
    )
    parser.add_argument("--skip-residual-reconfig", action="store_true")
    parser.add_argument("--center-k", action="store_true")
    parser.add_argument("--center-v", action="store_true")
    parser.add_argument(
        "--v-mean-correction", action="store_true", help="CPU-prepared V mean correction, fused FP32 epilogue"
    )
    parser.add_argument("--qkv-route", choices=("host", "device"), default="host")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--distribution", default="normal")
    parser.add_argument("--exp-quality", choices=("accurate", "cheap"), default="accurate")
    parser.add_argument("--exp-degree", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--native-exp", action="store_true")
    parser.add_argument("--destination", choices=("fp32", "fast_bf16", "main_bf16", "hybrid"), default="fp32")
    parser.add_argument("--safe-rescale", action="store_true")
    parser.add_argument("--device-preprocess", action="store_true")
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    opts = parser.parse_args()
    assert not (opts.round_p and opts.no_p_round)
    assert not opts.round_p or (opts.destination == "fp32" and opts.exp_quality == "accurate")
    assert opts.length % 512 == 0
    torch.set_num_threads(8)
    inputs = V2.V1_NUMERICS.FRONTIER.inputs_for(opts.length, opts.seed, opts.distribution)
    ref = repro.reference(*inputs)
    args = argparse.Namespace(**vars(opts))
    args.mode = "accurate" if opts.exp_quality == "accurate" else "fp32_hifi2_cheap"
    if opts.destination != "fp32":
        assert opts.variant in ("pervalue_rawp", "q7kv8_rawp")
        args.mode = "fast" if opts.destination == "fast_bf16" else "main"
    if opts.destination == "hybrid":
        args.mode, args.hybrid_block_pack = "hybrid", True
    args.q_repeats, args.k_chunks, args.distinct_kv = 1, opts.length // 512, True
    path = HERE / (opts.label + ".jsonl")
    assert not path.exists(), "Use a fresh label"
    device = ttnn.open_device(device_id=0, trace_region_size=1048576 if opts.iters else 0)
    try:
        record, actual = run(device, args, inputs, ref)
        record["kind"] = "fused_attention"
        with path.open("x") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
    finally:
        ttnn.close_device(device)
