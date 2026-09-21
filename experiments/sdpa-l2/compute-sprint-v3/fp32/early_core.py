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
SPRINT = "experiments/sdpa-l2/compute-sprint-v3/fp32/"


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
        q, k, v = make_inputs(repro, 256, 512 * args.k_chunks, args.seed, args.distribution)
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
    flags = {
        "baseline": ["SDPA_SPRINT_CANONICAL"],
        "copy": [],
        "identity4": ["SDPA_SPRINT_IDENTITY4"],
        "state_unpack": ["SDPA_SPRINT_STATE_UNPACK_BATCH"],
        "denom_pack": ["SDPA_SPRINT_DENOM_BLOCK_PACK"],
        "state_both": ["SDPA_SPRINT_IDENTITY4", "SDPA_SPRINT_STATE_UNPACK_BATCH"],
        "all": ["SDPA_SPRINT_IDENTITY4", "SDPA_SPRINT_STATE_UNPACK_BATCH", "SDPA_SPRINT_DENOM_BLOCK_PACK"],
        "c_refine": ["SDPA_SPRINT_C_REFINE"],
        "c_refine_state": ["SDPA_SPRINT_C_REFINE", "SDPA_SPRINT_IDENTITY4", "SDPA_SPRINT_STATE_UNPACK_BATCH"],
        "c_refine_hoist": ["SDPA_SPRINT_C_REFINE", "SDPA_SPRINT_C_REFINE_HOIST"],
        "c_refine_hoist_state": [
            "SDPA_SPRINT_C_REFINE",
            "SDPA_SPRINT_C_REFINE_HOIST",
            "SDPA_SPRINT_IDENTITY4",
            "SDPA_SPRINT_STATE_UNPACK_BATCH",
        ],
        "lazy_init": ["SDPA_SPRINT_LAZY_INIT"],
        "state_lazy": ["SDPA_SPRINT_IDENTITY4", "SDPA_SPRINT_STATE_UNPACK_BATCH", "SDPA_SPRINT_LAZY_INIT"],
        "c_refine_hoist_state_lazy": [
            "SDPA_SPRINT_C_REFINE",
            "SDPA_SPRINT_C_REFINE_HOIST",
            "SDPA_SPRINT_IDENTITY4",
            "SDPA_SPRINT_STATE_UNPACK_BATCH",
            "SDPA_SPRINT_LAZY_INIT",
        ],
    }
    flags["state_lazy_scan"] = flags["state_lazy"] + ["SDPA_SPRINT_MAX_SCAN"]
    flags["c_refine_hoist_state_lazy_scan"] = flags["c_refine_hoist_state_lazy"] + ["SDPA_SPRINT_MAX_SCAN"]
    defines.update({flag: "1" for flag in flags["state_lazy_scan" if args.variant == "D" else "c_refine_hoist_state_lazy_scan"]})
    source = SPRINT + "resident.cpp"
    if args.algorithm == "baseline":
        source = "experiments/sdpa-l2/compute-sprint-v1/fp32/resident.cpp"
        if args.variant == "D":
            source = "experiments/sdpa-l2/compute-sprint-v2/codegen/resident.cpp"
            defines["SDPA_CODEGEN_OPT"] = "2"
    else:
        if args.variant == "D":
            defines["SDPA_V3_O2"] = "1"
        if args.algorithm == "full_pv":
            defines["SDPA_V3_FULL_PV"] = "1"
        elif args.algorithm == "identity_fusion":
            defines["SDPA_V3_IDENTITY_FUSION"] = "1"
        elif args.algorithm == "fusion_batch":
            defines["SDPA_V3_IDENTITY_FUSION"] = "1"
            defines["SDPA_V3_FUSION_BATCH"] = "1"
        elif args.algorithm == "l1_inplace":
            defines["SDPA_V3_L1_INPLACE"] = "1"
        elif args.algorithm == "l1_early":
            source = SPRINT + "resident_early.cpp"
            defines["SDPA_V3_L1_INPLACE"] = "1"
            defines["SDPA_V3_L1_BOTH"] = "1"
        elif args.algorithm == "l1_both":
            defines["SDPA_V3_L1_INPLACE"] = "1"
            defines["SDPA_V3_L1_BOTH"] = "1"
        else:
            raise ValueError(args.algorithm)
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
                kernel_source=source,
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
            ROOT / "experiments/sdpa-l2/flux2-frontier-v1/device_attention.py",
            Path(__file__).resolve(),
            ROOT / "experiments/sdpa-l2/compute-sprint-v1/fp32/compute_streaming.hpp",
            ROOT / "experiments/sdpa-l2/compute-sprint-v1/fp32/resident.cpp",
            ROOT / "experiments/sdpa-l2/compute-sprint-v1/fp32/c_refine.hpp",
            ROOT / "experiments/sdpa-l2/compute-sprint-v2/codegen/resident.cpp",
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
    return record, actual, ref, (q, k, v)

def make_inputs(repro, q_length, k_length, seed, distribution):
    name = "normal" if distribution in ("changing_max", "transitions", "zero_v", "constant_325", "coherent", "cancellation", "tiny_update") else distribution
    q, k, v = repro.make_inputs(1, q_length, k_length, 128, seed, name)
    if distribution == "changing_max":
        q[..., 0] = 8
        k[..., 0] = (torch.arange(k_length) // 512 * 4).bfloat16()
    elif distribution == "zero_v":
        v.zero_()
    elif distribution == "transitions":
        q[..., 0] = 8
        k = k[..., :512, :].repeat(1, 1, k_length // 512, 1)
        k[..., 0] += (torch.arange(k_length) // 1024 * 4).bfloat16()
    elif distribution == "constant_325":
        v.fill_(3.25)
    elif distribution == "coherent":
        k = k[..., :512, :].repeat(1, 1, k_length // 512, 1)
        v = v[..., :512, :].repeat(1, 1, k_length // 512, 1)
    elif distribution == "cancellation":
        # Equal-score +/- V pairs, with a small representable asymmetry.
        k[..., 1::2, :] = k[..., 0::2, :]
        v[..., 1::2, :] = -v[..., 0::2, :]
        v[..., -1, :] = (v[..., -1, :].float() + 0.03125).bfloat16()
    elif distribution == "tiny_update":
        # Uniform attention: large first numerator near an output BF16 tie,
        # followed by small products that can be lost if merged into DST early.
        q.zero_()
        k.zero_()
        v.fill_(2**-20)
        v[..., :256, :] = 1.0
        v[..., 256:512, :] = 1.0078125
    return q, k, v
