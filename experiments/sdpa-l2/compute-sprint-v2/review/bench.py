# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pinned A/B compute-only benchmark; caller owns exclusive device locking."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import struct
import time

import torch
import ttnn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/resident/"
DISTINCT_READER = "experiments/sdpa-l2/hybrid-mixed-v1/reader_distinct.cpp"
OWN = "experiments/sdpa-l2/compute-sprint-v2/review/"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANON = load("bf16_sprint_canonical", ROOT / "experiments/sdpa-l2/flux2-frontier-v1/device_attention.py")
REPRO = load("bf16_sprint_inputs", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")


def build(device, args, tensors, candidate):
    fp32, compensated, numeric, fidelity = CANON.recipe(args.variant)
    assert not fp32 and fidelity == ttnn.MathFidelity.HiFi2
    defines = dict(numeric)
    implementation = {name: "1" for name in ("SDPA_BF16_BLOCK_STATE", "SDPA_BF16_CORRECTION_REUSE", "SDPA_BF16_CORRECTION_FENCE")}
    if candidate == "correction_cache":
        implementation["SDPA_REVIEW_CANDIDATE"] = "1"
    if candidate == "pack_overlap":
        implementation["SDPA_REVIEW_PACK_OVERLAP"] = "1"
    if candidate == "identity":
        implementation["SDPA_REVIEW_IDENTITY"] = "1"
    if candidate == "identity_fastguard":
        implementation["SDPA_REVIEW_IDENTITY_FASTGUARD"] = "1"
    if candidate == "identity_early":
        implementation["SDPA_REVIEW_IDENTITY_EARLY"] = "1"
    if candidate == "phase_profile":
        implementation["SDPA_REVIEW_PHASE_PROFILE"] = "1"
    defines.update(implementation)
    assert {k: v for k, v in defines.items() if k not in implementation} == numeric
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    stride = 2 if compensated else 1
    counts = [(0, 64), (1, 128), (2, 128), (3, 1), (4, 1), (5, 1), (6, 128),
              (8, 32 * stride), (9, 32 * stride), (10, 8), (11, 8),
              (12, 8 * stride), (13, 8 * stride), (14, 8), (16, 16)]
    cbs = [ttnn.CBDescriptor(total_size=n * 2048, core_ranges=grid,
                            format_descriptors=[ttnn.CBFormatDescriptor(
                                buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)]) for i, n in counts]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 256, 128]), ttnn.bfloat16,
                                       ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    reader, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader[0][0] = [t.buffer_address() for t in tensors]
    writer[0][0] = [out.buffer_address()]
    reader_cta = [args.q_repeats, args.k_chunks, 2]
    for tensor in tensors:
        reader_cta += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    descriptor = ttnn.ProgramDescriptor(cbs=cbs, semaphores=[], kernels=[
        ttnn.KernelDescriptor(kernel_source=DISTINCT_READER if args.distinct_kv else PREFIX + "reader.cpp",
                              core_ranges=grid, compile_time_args=reader_cta, runtime_args=reader,
                              config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=PREFIX + "writer.cpp", core_ranges=grid,
                              compile_time_args=[args.q_repeats] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                              runtime_args=writer, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=("experiments/sdpa-l2/compute-sprint-v1/bf16/compute.cpp" if candidate == "canonical" else OWN + "resident.cpp"),
                              core_ranges=grid, compile_time_args=[args.q_repeats, args.k_chunks,
                                  struct.unpack("I", struct.pack("f", 1 / math.sqrt(128)))[0]],
                              defines=list(defines.items()), config=ttnn.ComputeConfigDescriptor(
                                  math_fidelity=fidelity, fp32_dest_acc_en=False,
                                  dst_full_sync_en=False, math_approx_mode=True)),
    ])
    def invoke():
        ttnn.generic_op(tensors + [out], descriptor)
    return out, invoke, dict(numeric_defines=numeric, implementation_defines=implementation,
                             cb_counts=counts, kv_slots=2, math_fidelity=str(fidelity),
                             fp32_dest_acc_en=False, math_approx_mode=True, dst_full_sync_en=False)


def run(args):
    torch.set_num_threads(8)
    n = 512 * args.k_chunks if args.distinct_kv else 512
    q, k, v = REPRO.make_inputs(1, 256, n, 128, args.seed,
                               "normal" if args.distribution in ("growing_max", "identity_transitions") else args.distribution)
    if args.distribution == "growing_max":
        assert args.distinct_kv
        factors = torch.linspace(0.25, 4.0, args.k_chunks).repeat_interleave(512)
        k = (k.float() * factors.reshape(1, 1, n, 1)).bfloat16()
    if args.distribution == "identity_transitions":
        assert args.distinct_kv and args.k_chunks % 2 == 0 and args.k_chunks <= 16
        factors = (2.0 ** (torch.arange(args.k_chunks)//2)).repeat_interleave(512)
        k = (k[:,:,:512,:].repeat(1,1,args.k_chunks,1).float()
             * factors.reshape(1,1,n,1)).bfloat16()
    reference = REPRO.reference(q, k, v)
    score_sample = q[..., ::32, :].double() @ k.double().transpose(-1, -2)
    chunk_max = score_sample.reshape(1, 1, 8, -1, 512).amax(-1)
    running_max = chunk_max.cummax(-1).values
    changed_max = (running_max[..., 1:] > running_max[..., :-1]).sum(-1)
    output_path = HERE / (args.label + ".json")
    assert not output_path.exists(), "Use a fresh label"
    device = ttnn.open_device(device_id=args.device_id, trace_region_size=16777216 if args.iters else 0)
    traces = {}
    try:
        tensors = [ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT) for x in (q, k, v)]
        variants = ["canonical", "disabled", args.candidate]
        records, outputs = {}, {}
        for candidate in variants:
            out, invoke, provenance = build(device, args, tensors, candidate)
            invoke()
            actual = ttnn.to_torch(out)
            assert torch.isfinite(actual).all()
            outputs[candidate] = actual
            flat = actual.double().flatten()
            ref = reference.double().flatten()
            records[candidate] = dict(**provenance,
                output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
                l2_pct=100 * ((flat-ref).norm()/ref.norm()).item(),
                pcc=torch.corrcoef(torch.stack((flat, ref)))[0, 1].item(), times_ms=[])
            if args.iters:
                trace = ttnn.begin_trace_capture(device, cq_id=0)
                invoke()
                ttnn.end_trace_capture(device, trace, cq_id=0)
                traces[candidate] = (trace, out)
        assert torch.equal(outputs["canonical"].view(torch.uint16), outputs["disabled"].view(torch.uint16)), "Disabled private copy differs from canonical"
        for candidate in variants:
            a, b = outputs[candidate], outputs["canonical"]
            records[candidate]["bitwise_equal_canonical"] = bool(torch.equal(a.view(torch.uint16), b.view(torch.uint16)))
            records[candidate]["unequal_elements"] = int((a.view(torch.uint16) != b.view(torch.uint16)).sum())
            records[candidate]["max_abs_change"] = float((a.float()-b.float()).abs().max())
        if args.iters:
            for i in range(args.warmup + args.iters):
                order = variants if i % 2 == 0 else list(reversed(variants))
                for candidate in order:
                    trace, out = traces[candidate]
                    start = time.perf_counter()
                    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                    elapsed = (time.perf_counter()-start)*1000
                    if i >= args.warmup:
                        records[candidate]["times_ms"].append(elapsed)
            for candidate, (_, out) in traces.items():
                actual_trace = ttnn.to_torch(out)
                records[candidate]["trace_equal"] = bool(torch.equal(
                    outputs[candidate].view(torch.uint16), actual_trace.view(torch.uint16)))
                assert records[candidate]["trace_equal"], "Eager/trace raw-bit mismatch"
        flops = 4*256*512*128*args.q_repeats*args.k_chunks
        for record in records.values():
            record["median_ms"] = statistics.median(record["times_ms"]) if record["times_ms"] else None
            record["tflops_per_core"] = flops/(record["median_ms"]*1e9) if record["median_ms"] else None
        sources = [Path(__file__).resolve(), HERE/"resident.cpp", HERE/"profile_streaming.hpp",
                   ROOT/"experiments/sdpa-l2/compute-sprint-v1/bf16/compute.cpp",
                   ROOT/"experiments/sdpa-l2/compute-sprint-v1/bf16/streaming_B.hpp",
                   ROOT/"experiments/sdpa-l2/compute-sprint-v1/bf16/compensated_block.hpp",
                   ROOT/"experiments/sdpa-l2/compute-sprint-v1/bf16/compensated_reuse.hpp",
                   ROOT/"experiments/sdpa-l2/compute-sprint-v1/lowp/block_state/compensated_block.hpp",
                   ROOT/"experiments/sdpa-l2/compute-sprint-v1/lowp/correction_reuse/compensated_reuse.hpp",
                   ROOT/"experiments/sdpa-l2/compute-sprint-v2/compensated/correction_cache/compute_streaming.hpp",
                   ROOT/"experiments/sdpa-l2/compute-sprint-v2/compensated/correction_cache/retain_release.hpp",
                   ROOT/PREFIX/"reader.cpp",ROOT/DISTINCT_READER,ROOT/PREFIX/"writer.cpp",
                   ROOT/"experiments/sdpa-l2/flux2-frontier-v1/device_attention.py",
                   ROOT/"experiments/sdpa-l2/bfp4-lofi-v2/fast_correction.hpp",
                   ROOT/"experiments/sdpa-l2/bfp4-lofi-v2/exp_refiner.hpp",
                   ROOT/"tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h",
                   ROOT/"tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h"]
        sources += list((ROOT/"experiments/sdpa-l2/compute-sprint-v2/compensated/pack_overlap").glob("*.hpp"))
        sources += list((ROOT/"experiments/sdpa-l2/compute-sprint-v2/compensated/identity").glob("*.hpp"))
        sources += list((ROOT/"experiments/sdpa-l2/compute-sprint-v2/compensated/identity_fastguard").glob("*.hpp"))
        sources += list((ROOT/"experiments/sdpa-l2/compute-sprint-v2/compensated/identity_early").glob("*.hpp"))
        for snapshot in ("single-core-resident-v1/main", "bf16-denom-pair-v3/candidate"):
            sources += list((ROOT/"experiments/sdpa-l2"/snapshot).rglob("*.h"))
            sources += list((ROOT/"experiments/sdpa-l2"/snapshot).rglob("*.hpp"))
        report = dict(arguments=vars(args), shape=dict(q=256,k=512,d=128,cores=1), useful_flops=flops,
                      sampled_query_max_changes_after_first_chunk=changed_max.flatten().tolist(),
                      warning="Distinct-KV correctness/DM test" if args.distinct_kv else "Resident repeated KV, no recurring input DM",
                      results=records, source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        output_path.write_text(json.dumps(report, indent=2)+"\n")
        print("RESULT "+json.dumps(report), flush=True)
        assert records[args.candidate]["bitwise_equal_canonical"], "Candidate numerical mismatch; see saved report"
    finally:
        for trace, _ in traces.values():
            ttnn.release_trace(device, trace)
        ttnn.close_device(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("B",), required=True)
    parser.add_argument("--candidate", choices=("correction_cache", "phase_profile", "pack_overlap", "identity", "identity_fastguard", "identity_early"), default="correction_cache")
    parser.add_argument("--label", required=True)
    parser.add_argument("--distinct-kv", action="store_true")
    parser.add_argument("--q-repeats", type=int, default=16)
    parser.add_argument("--k-chunks", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1236)
    parser.add_argument("--distribution", default="normal")
    parser.add_argument("--device-id", type=int, default=0)
    run(parser.parse_args())
