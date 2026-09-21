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
ROOT = HERE.parents[3]
PREFIX = "experiments/sdpa-l2/hybrid-mixed-v1/"
SPRINT = "experiments/sdpa-l2/compute-sprint-v2/fp32/"


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
    # Canonical recipe is the sole numeric authority.
    recipe_spec = importlib.util.spec_from_file_location(
        "sprint_device_attention", ROOT / "experiments/sdpa-l2/flux2-frontier-v1/device_attention.py"
    )
    recipe_mod = importlib.util.module_from_spec(recipe_spec)
    recipe_spec.loader.exec_module(recipe_mod)
    fp32, fast_recipe, defines, fidelity = recipe_mod.recipe(args.variant)
    assert fp32 and not fast_recipe
    base_flags = [
        "SDPA_SPRINT_IDENTITY4",
        "SDPA_SPRINT_STATE_UNPACK_BATCH",
        "SDPA_SPRINT_LAZY_INIT",
        "SDPA_SPRINT_MAX_SCAN",
    ]
    if args.variant == "C":
        base_flags += ["SDPA_SPRINT_C_REFINE", "SDPA_SPRINT_C_REFINE_HOIST"]
    flags = {
        "baseline": ["SDPA_V2_BASELINE"],
        "copy": [],
        "statepack": ["SDPA_V2_STATE_BLOCKPACK"],
        "bulkidentity": ["SDPA_V2_BULK_IDENTITY"],
        "statepack_bulkidentity": ["SDPA_V2_STATE_BLOCKPACK", "SDPA_V2_BULK_IDENTITY"],
        "phase_profile": ["SDPA_UTIL_PROFILE"],
        "identitybatch": ["SDPA_V2_STATE_BLOCKPACK", "SDPA_V2_BULK_IDENTITY", "SDPA_V2_IDENTITY_BATCH"],
        "score_srca": ["SDPA_V2_SCORE_SRCA"],
        "state_srca": ["SDPA_V2_STATE_SRCA"],
        "unary_srca": ["SDPA_V2_SCORE_SRCA", "SDPA_V2_STATE_SRCA"],
        "zeroacc_half": ["SDPA_V2_ZEROACC_HALF"],
        "qk22": ["SDPA_V2_QK22", "SDPA_FP32_QK_WIDTH2"],
        "timeline": ["SDPA_V2_TIMELINE"],
        "identity_timeline": ["SDPA_V2_TIMELINE", "SDPA_V2_STATE_BLOCKPACK", "SDPA_V2_BULK_IDENTITY", "SDPA_V2_IDENTITY_BATCH"],
    }
    defines.update({flag: "1" for flag in base_flags + flags[args.candidate]})
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity, fp32_dest_acc_en=True, dst_full_sync_en=False, math_approx_mode=True
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
                kernel_source=SPRINT + "resident.cpp",
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
        + list((ROOT / SPRINT).glob("*.cpp"))
        + list((ROOT / SPRINT).glob("*.hpp"))
        + list((ROOT / "experiments/sdpa-l2/bfp4-lofi-v2").glob("*.hpp"))
        + [
            ROOT / "experiments/sdpa-l2/bfp4-lofi-v2/streaming/compute_streaming.hpp",
            ROOT / "experiments/sdpa-l2/compute-sprint-v1/fp32/compute_streaming.hpp",
            ROOT / "experiments/sdpa-l2/compute-sprint-v1/fp32/c_refine.hpp",
            ROOT / "experiments/sdpa-l2/flux2-frontier-v1/device_attention.py",
            Path(__file__).resolve(),
        ]
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
    # Qualification uses baseline/candidate bitwise equality.
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
            assert torch.equal(actual.view(torch.uint16), ttnn.to_torch(out).view(torch.uint16))
        finally:
            ttnn.release_trace(device, trace)
    flops = 4 * 256 * 512 * 128 * args.q_repeats * args.k_chunks
    median = statistics.median(times) if times else None
    record = dict(
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

    for tensor in tensors + [out]:
        ttnn.deallocate(tensor)
    return record, actual


CANDIDATES = ["copy", "statepack", "bulkidentity", "statepack_bulkidentity", "phase_profile", "identitybatch", "score_srca", "state_srca", "unary_srca", "zeroacc_half", "timeline", "identity_timeline", "qk22"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--variants", default="D,C")
    parser.add_argument("--candidates", default="copy,statepack,bulkidentity,statepack_bulkidentity")
    parser.add_argument("--q-repeats", type=int, default=8)
    parser.add_argument("--k-chunks", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1236)
    parser.add_argument(
        "--distribution",
        default="normal",
        choices=["normal", "constant_v", "scaled_qk", "outliers", "common_q", "common_k", "common_v", "uniform"],
    )
    parser.add_argument("--distinct-kv", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    assert args.q_repeats > 0 and args.k_chunks > 0
    assert args.distinct_kv or args.distribution in ("normal", "constant_v")
    output = HERE / (args.label + ".json")
    assert not output.exists(), "Use a fresh label"
    records = []
    variants = args.variants.split(",")
    candidates = args.candidates.split(",")
    assert set(variants) <= {"C", "D"} and set(candidates) <= set(CANDIDATES)
    assert not ("D" in variants and any(c.startswith("c_refine") for c in candidates))
    device = ttnn.open_device(device_id=0, trace_region_size=16777216 if args.iters else 0)
    try:
        for variant in variants:
            # Eager source-pinned baseline oracle independent of timed order.
            oracle_args = argparse.Namespace(
                **vars(args),
                variant=variant,
                candidate="baseline",
                mode="accurate" if variant == "D" else "qk4_pv2",
                hybrid_block_pack=False,
            )
            oracle_args.label = args.label + "-" + variant + "-oracle"
            oracle_args.iters = 0
            _, oracle = run(device, oracle_args)
            order = ["baseline"] + candidates
            if args.reverse:
                order.reverse()
            for candidate in order:
                item_args = argparse.Namespace(
                    **vars(args),
                    variant=variant,
                    candidate=candidate,
                    mode="accurate" if variant == "D" else "qk4_pv2",
                    hybrid_block_pack=False,
                )
                item_args.label = args.label + "-" + variant + "-" + candidate
                record, actual = run(device, item_args)
                record["baseline_bitwise_equal"] = bool(
                    torch.equal(actual.view(torch.uint16), oracle.view(torch.uint16))
                )
                record["baseline_mismatched_values"] = int(
                    (actual.view(torch.uint16) != oracle.view(torch.uint16)).sum().item()
                )
                record["baseline_max_abs_diff"] = float((actual.float() - oracle.float()).abs().max())
                records.append(record)
                torch.save(actual, HERE / (item_args.label + ".pt"))
                output.write_text(json.dumps(records, indent=2) + "\n")
                print("RESULT " + json.dumps(record), flush=True)
                if not record["baseline_bitwise_equal"]:
                    raise AssertionError("Scheduling candidate changed output: " + candidate)
    finally:
        ttnn.close_device(device)
