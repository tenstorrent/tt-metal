# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated BF16/FP32-state and QK4/PV2 streaming experiments, Q256/K512/D128."""

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
PREFIX = "experiments/sdpa-l2/hybrid-mixed-v1/"


def run(device, args):
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
    torch.manual_seed(args.seed)
    q = torch.randn(1, 1, 256, 128).bfloat16()
    k = torch.randn(1, 1, 512, 128).bfloat16()
    v = torch.randn(1, 1, 512, 128).bfloat16()
    if args.distribution == "constant_v":
        v.fill_(1)
    # Duplicate resident K/V chunks leave the exact normalized attention unchanged.
    ref = torch.softmax(q.double() @ k.double().transpose(-2, -1) / math.sqrt(128), dim=-1) @ v.double()
    if args.distinct_kv:
        spec = importlib.util.spec_from_file_location(
            "sdpa_repro", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py"
        )
        repro = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(repro)
        torch.set_num_threads(8)
        q, k, v = repro.make_inputs(1, 256, 512 * args.k_chunks, 128, args.seed, args.distribution)
        ref = repro.reference(q, k, v)
    tensors = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in (q, k, v)]
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 256, 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    kv_slots = 1 if accurate else 2
    fp = ttnn.float32 if accurate else ttnn.bfloat16
    page = 4096 if accurate else 2048
    specs = [
        (0, 64, 2048, ttnn.bfloat16),
        (1, 64 * kv_slots, 2048, ttnn.bfloat16),
        (2, 64 * kv_slots, 2048, ttnn.bfloat16),
        (3, 1, 2048, ttnn.bfloat16),
        (4, 1, 2048, ttnn.bfloat16),
        (5, 1, page, fp),
        (6, 128, page, fp),
        (8, 32 * (2 if fast else 1), page, fp),
        (9, 32 * (2 if fast else 1), page, fp),
        (10, 8, 2048, ttnn.bfloat16),
        (11, 8, 2048, ttnn.bfloat16),
        (12, 8 * (2 if fast else 1), page, fp),
        (13, 8 * (2 if fast else 1), page, fp),
        (14, 8, page, fp),
        (16, 8 if accurate else 16, 2048, ttnn.bfloat16),
    ]
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
        "STATS_GRANULARITY": "4" if accurate else "8",
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
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4 if args.mode == "accurate" else ttnn.MathFidelity.HiFi2,
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
                kernel_source=PREFIX + ("reader_distinct.cpp" if args.distinct_kv else "reader.cpp"),
                core_ranges=grid,
                compile_time_args=reader_cta,
                runtime_args=reader,
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
                    struct.unpack("I", struct.pack("f", 1 / math.sqrt(128)))[0],
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
        + [Path(__file__).resolve()]
    )
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
    assert torch.isfinite(actual).all()
    l2 = 100 * ((actual.double() - ref).norm() / ref.norm()).item()
    assert l2 < (0.5 if args.mode == "accurate" and not args.distinct_kv else 100), l2
    if hybrid and args.distribution == "normal":
        assert l2 < 5, l2
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
    flops = 4 * 256 * 512 * 128 * args.q_repeats * args.k_chunks
    median = statistics.median(times) if times else None
    return dict(
        **vars(args),
        cores=1,
        q_chunk=256,
        k_chunk=512,
        dim=128,
        kv_slots=kv_slots,
        cb_bytes=sum(n * size for _, n, size, _ in specs),
        useful_flops=flops,
        median_ms=median,
        trace_replay_ms=times,
        tflops_per_core=flops / (median * 1e9) if median else None,
        l2_pct=l2,
        pcc=torch.corrcoef(torch.stack((actual.double().flatten(), ref.flatten())))[0, 1].item(),
        full_output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
        trace_equal=bool(times),
        warning=(
            "Distinct K/V correctness test, not a no-DM throughput measurement"
            if args.distinct_kv
            else "Repeated resident Q/K/V, not general-input accuracy qualification or measured chip throughput"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "fast",
            "accurate",
            "fp32_hifi2",
            "fp32_hifi2_cheap",
            "qk4_pv2",
            "qk4_pv2_fullsub",
            "qk2_pv2_fullsub",
            "hybrid",
        ],
        required=True,
    )
    parser.add_argument("--q-repeats", type=int, default=8)
    parser.add_argument("--k-chunks", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1236)
    parser.add_argument(
        "--distribution",
        choices=["normal", "constant_v", "scaled_qk", "outliers", "common_q", "common_k", "common_v", "uniform"],
        default="normal",
    )
    parser.add_argument("--distinct-kv", action="store_true")
    parser.add_argument("--hybrid-block-pack", action="store_true")
    args = parser.parse_args()
    result = HERE / (args.label + ".json")
    assert not result.exists(), "Use a fresh label"
    assert args.q_repeats > 0 and args.k_chunks > 0
    assert args.distinct_kv or args.distribution in ("normal", "constant_v")
    device = ttnn.open_device(device_id=0, trace_region_size=16777216 if args.iters else 0)
    try:
        record = run(device, args)
        result.write_text(json.dumps(record, indent=2) + "\n")
        print("RESULT " + json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)
