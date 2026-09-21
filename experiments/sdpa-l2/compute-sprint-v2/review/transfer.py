# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Canonical fullchip dataflow, distinct Q jobs per core; exact output checks."""
import argparse
import hashlib
import json
import torch
import ttnn
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location("review_v1_bench", ROOT/"experiments/sdpa-l2/compute-sprint-v1/bf16/bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.HERE = HERE
bench.OWN = str(HERE.relative_to(ROOT)) + "/"
V1_FLAGS = ["SDPA_BF16_BLOCK_STATE", "SDPA_BF16_CORRECTION_REUSE", "SDPA_BF16_CORRECTION_FENCE"]

FLAGS = {"correction_cache": ["SDPA_REVIEW_CANDIDATE"],
         "identity": ["SDPA_REVIEW_IDENTITY"],
         "identity_fastguard": ["SDPA_REVIEW_IDENTITY_FASTGUARD"],
         "identity_early": ["SDPA_REVIEW_IDENTITY_EARLY"],
         "pack_overlap": ["SDPA_REVIEW_PACK_OVERLAP"]}

def main(args):
    torch.set_num_threads(8)
    output = bench.HERE / (args.label + ".json")
    assert not output.exists()
    original = ttnn.KernelDescriptor
    mode = "canonical"
    descriptors = []
    def construct(*pos, **kwargs):
        if kwargs["kernel_source"].endswith("/compute.cpp"):
            kwargs["defines"] = list(kwargs["defines"]) + [(flag, "1") for flag in V1_FLAGS]
            kwargs["kernel_source"] = "experiments/sdpa-l2/compute-sprint-v1/bf16/fullchip_compute.cpp"
        if mode != "canonical" and kwargs["kernel_source"].endswith("/fullchip_compute.cpp"):
            kwargs["kernel_source"] = bench.OWN + "compute.cpp"
            if mode != "disabled":
                kwargs["defines"] = list(kwargs["defines"]) + [(flag, "1") for flag in FLAGS[args.candidate]]
        descriptors.append({key: kwargs.get(key) for key in ("kernel_source", "defines", "compile_time_args")})
        return original(*pos, **kwargs)
    records = []
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        for distribution in args.distributions.split(","):
            q,k,v = bench.REPRO.make_inputs(1,args.q_length,args.k_length,128,args.seed,
                                             "normal" if distribution in ("growing_max", "identity_transitions") else distribution)
            if distribution == "growing_max":
                factors = torch.linspace(0.25,4,args.k_length//512).repeat_interleave(512)
                k = (k.float()*factors.reshape(1,1,args.k_length,1)).bfloat16()
            if distribution == "identity_transitions":
                assert args.k_length % 1024 == 0 and args.k_length <= 8192
                # Identical K pairs separated by exact power-of-two growth:
                # 1,1,2,2,4,4,... . V remains independently sampled per token.
                factors = (2.0 ** (torch.arange(args.k_length//512)//2)).repeat_interleave(512)
                k = (k[:,:,:512,:].repeat(1,1,args.k_length//512,1).float()
                     * factors.reshape(1,1,args.k_length,1)).bfloat16()
            inputs = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in (q,k,v)]
            oracle = None
            for mode in ("canonical", "disabled", args.candidate):
                descriptors.clear()
                ttnn.KernelDescriptor = construct
                try:
                    def invoke():
                        return bench.CANON.attention(device,*inputs,args.variant,max_cores=args.cores,
                                                     q_chunk_size=256,k_chunk_size=512,reader_barrier_tiles=2)
                    out = invoke()
                    actual = ttnn.to_torch(out)
                    ttnn.deallocate(out)
                    if oracle is None:
                        oracle = actual.clone()
                    equal = torch.equal(actual.view(torch.uint16),oracle.view(torch.uint16))
                    trace = ttnn.begin_trace_capture(device,cq_id=0)
                    traced_out = invoke()
                    ttnn.end_trace_capture(device,trace,cq_id=0)
                    try:
                        for _ in range(2):
                            ttnn.execute_trace(device,trace,cq_id=0,blocking=True)
                            assert torch.equal(actual.view(torch.uint16),ttnn.to_torch(traced_out).view(torch.uint16))
                    finally:
                        ttnn.release_trace(device,trace)
                    ttnn.deallocate(traced_out)
                finally:
                    ttnn.KernelDescriptor = original
                records.append(dict(distribution=distribution,mode=mode,bitwise_equal=bool(equal),
                    unequal_elements=int((actual.view(torch.uint16)!=oracle.view(torch.uint16)).sum()),
                    max_abs_change=float((actual.float()-oracle.float()).abs().max()),
                    output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
                    descriptors=list(descriptors),trace_replays=2))
                sources = set(bench.HERE.glob("*.hpp")) | {bench.HERE/"compute.cpp",bench.HERE/"transfer.py",
                    bench.ROOT/"experiments/sdpa-l2/compute-sprint-v1/bf16/streaming_B.hpp",
                    bench.ROOT/"experiments/sdpa-l2/compute-sprint-v2/compensated/correction_cache/compute_streaming.hpp",
                    bench.ROOT/"experiments/sdpa-l2/compute-sprint-v2/compensated/correction_cache/retain_release.hpp",
                    bench.ROOT/"experiments/sdpa-l2/flux2-frontier-v1/device_attention.py",
                    bench.ROOT/"experiments/sdpa-l2/bfp4-lofi-v2/fast_correction.hpp",
                    bench.ROOT/"experiments/sdpa-l2/bfp4-lofi-v2/exp_refiner.hpp",
                    bench.ROOT/"tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h"}
                sources |= {bench.ROOT/d["kernel_source"] for r in records for d in r["descriptors"]}
                sources |= set((bench.ROOT/"experiments/sdpa-l2/compute-sprint-v2/compensated/pack_overlap").glob("*.hpp"))
                sources |= set((bench.ROOT/"experiments/sdpa-l2/compute-sprint-v2/compensated/identity").glob("*.hpp"))
                sources |= set((bench.ROOT/"experiments/sdpa-l2/compute-sprint-v2/compensated/identity_fastguard").glob("*.hpp"))
                sources |= set((bench.ROOT/"experiments/sdpa-l2/compute-sprint-v2/compensated/identity_early").glob("*.hpp"))
                sources |= {bench.ROOT/path for path in (
                    "experiments/sdpa-l2/compute-sprint-v1/bf16/bench.py",
                    "experiments/sdpa-l2/compute-sprint-v1/bf16/compensated_block.hpp",
                    "experiments/sdpa-l2/compute-sprint-v1/bf16/compensated_reuse.hpp",
                    "experiments/sdpa-l2/compute-sprint-v1/lowp/block_state/compensated_block.hpp",
                    "experiments/sdpa-l2/compute-sprint-v1/lowp/correction_reuse/compensated_reuse.hpp",
                    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h",
                    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h",
                    "tt_metal/hw/inc/api/compute/tile_move_copy.h",
                    "tt_metal/hw/inc/api/compute/reg_api.h",
                    "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py",
                )}
                for snapshot in ("single-core-resident-v1/main", "bf16-denom-pair-v3/candidate"):
                    for suffix in ("*.h", "*.hpp"):
                        sources |= set((bench.ROOT/"experiments/sdpa-l2"/snapshot).rglob(suffix))
                report = dict(arguments=vars(args),results=records,source_sha256={
                    str(p.relative_to(bench.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(sources)})
                output.write_text(json.dumps(report,indent=2)+"\n")
                print("RESULT "+json.dumps(records[-1]),flush=True)
                assert equal, "Candidate fullchip mismatch"
            for tensor in inputs:
                ttnn.deallocate(tensor)
    finally:
        ttnn.KernelDescriptor = original
        ttnn.close_device(device)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--variant",choices=("B",),required=True)
    p.add_argument("--candidate",choices=tuple(FLAGS),required=True)
    p.add_argument("--label",required=True)
    p.add_argument("--q-length",type=int,default=2048)
    p.add_argument("--k-length",type=int,default=8192)
    p.add_argument("--cores",type=int,default=2)
    p.add_argument("--seed",type=int,default=1237)
    p.add_argument("--distributions",default="normal,growing_max,scaled_qk,outliers,common_q,common_k,common_v,constant_v")
    main(p.parse_args())
