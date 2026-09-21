# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""E/G grouped-compensation numerical/performance prototype; exclusive device access required.

Canonical numerical authority is flux2-frontier-v1/device_attention.py.
Resident mode has no recurring input DM; distinct mode retains the original recurring-DM reader.
No changes to Q256/K512/D128, input slots, or numerical preparation are allowed.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import re
import statistics
import struct
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESEARCH = ROOT / "experiments/sdpa-l2/bfp4-lofi-v2"
FROZEN = ROOT / "experiments/sdpa-l2/bf16-denom-pair-v3/candidate"
PREFIX = str(HERE.relative_to(ROOT)) + "/"
V1 = ROOT / "experiments/sdpa-l2/compute-sprint-v1/lowp"
V2 = ROOT / "experiments/sdpa-l2/compute-sprint-v2/compensated"
BASELINE = V2 / "identity_early/compute_streaming.hpp"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANONICAL = load("sprint_lowp_canonical", HERE.parents[1] / "flux2-frontier-v1/device_attention.py")
REPRO = load("sprint_lowp_repro", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")
NUMERICS = load("sprint_v3_numerics", HERE.parent / "numerics.py")


def tensor_hash(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def bits_equal(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(
        a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def source_hashes(candidate=None):
    # Selected implementation, preprocessing and oracle dependencies, not a
    # claim to hash the compiler/firmware closure.
    files = [Path(__file__).resolve(), HERE / "compute.cpp", Path(CANONICAL.__file__), Path(REPRO.__file__)]
    files += [RESEARCH / name for name in (
        "preprocess.py", "bfp4_round.py", "numerics.py", "safe_rescale.hpp",
        "fast_correction.hpp", "exp_refiner.hpp", "exp_native.hpp", "round_p.hpp",
        "fullchip/compute.cpp", "fullchip/reader.cpp", "fullchip/reader_chain.cpp", "fullchip/writer.cpp",
        "resident/reader.cpp", "resident/writer.cpp",
    )]
    for directory in (RESEARCH / "preprocess", RESEARCH / "bfp4_round", FROZEN):
        files += [p for p in directory.rglob("*") if p.suffix in (".cpp", ".h", ".hpp")]
    files += [RESEARCH.parent / "bfp4-lofi-v1" / name for name in ("probe.py", "numerics.py")]
    files += [ROOT / path for path in (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_streaming_qktv.hpp",
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/q_chunk_remapping.hpp",
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp",
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp",
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp",
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp",
        "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp",
        "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp",
        "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl",
        "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp",
        "tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h",
        "tt_metal/hw/inc/api/compute/experimental/matmul_custom.h",
        "tt_metal/hw/inc/api/compute/experimental/sdpa_sub_custom.h",
        "tt_metal/hw/inc/api/compute/bcast.h",
        "tt_metal/hw/inc/api/compute/tile_move_copy.h",
        "tt_metal/hw/inc/api/compute/reg_api.h",
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h",
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h",
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h",
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h",
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h",
        "tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h",
        "tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h",
        "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h",
        "tt_metal/hw/inc/api/compute/eltwise_unary/exp.h",
        "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h",
        "tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_pack.h",
        "tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_unpack.h",
        "tt_metal/hw/inc/api/dataflow/circular_buffer.h",
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h",
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h",
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h",
    )]
    if candidate or BASELINE:
        pending, visited = [BASELINE] + ([candidate] if candidate else []), set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            files.append(current)
            for include in re.findall(r'^\s*#include\s+"([^"]+)"',current.read_text(),re.MULTILINE):
                for target in ((current.parent/include).resolve(),(ROOT/include).resolve()):
                    if target.is_file() and (target.is_relative_to(HERE) or target.is_relative_to(V1) or target.is_relative_to(V2)):
                        pending.append(target)
                        break
    files.append(HERE.parent / "numerics.py")
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


def build(device, prepared, variant, *, resident, jobs, k_chunks, candidate=None):
    assert variant in ("E", "G") and jobs > 0 and k_chunks > 0
    fp32, fast, defines, fidelity = CANONICAL.recipe(variant)
    assert not fp32 and fast and fidelity == ttnn.MathFidelity.LoFi
    defines = dict(defines)
    defines["SDPA_K_CHUNK_TILES"] = "16"
    defines["SDPA_SPRINT_CANDIDATE_HEADER"] = '"' + str((candidate or BASELINE).relative_to(ROOT)) + '"'
    kv_type, kv_bytes = (ttnn.bfloat8_b, 1088) if variant == "E" else (ttnn.bfloat4_b, 576)
    assert [x.get_dtype() for x in prepared] == [ttnn.bfloat16, kv_type, kv_type]
    assert tuple(prepared[0].shape) == (1, 1, 256 if resident else 256*jobs, 128)
    assert tuple(prepared[1].shape) == (1, 1, 512 if resident else 512 * k_chunks, 128)
    assert tuple(prepared[2].shape) == tuple(prepared[1].shape)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    bf = ttnn.bfloat16
    # Exactly the canonical Q256/K512 E/G CB sizes and formats, including the
    # second Q slot and BOTH K/V input slots in no-DM mode.
    specs = [(0,64,2048,bf), (1,128,kv_bytes,kv_type), (2,128,kv_bytes,kv_type),
             (3,1,2048,bf), (4,1,2048,bf), (5,1,2048,bf), (6,128,2048,bf),
             (8,64,2048,bf), (9,64,2048,bf), (10,8,2048,bf), (11,8,2048,bf),
             (12,16,2048,bf), (13,16,2048,bf), (14,8,2048,bf), (16,16,2048,bf)]
    cbs = [ttnn.CBDescriptor(total_size=n*size, core_ranges=grid,
              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i,data_format=fmt,page_size=size)])
           for i,n,size,fmt in specs]
    out = ttnn.allocate_tensor_on_device(prepared[0].shape, bf, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    read, write = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    read[0][0] = [x.buffer_address() for x in prepared] + ([] if resident else [0,jobs])
    write[0][0] = [out.buffer_address()] + ([] if resident else [0,jobs])
    read_cta = [jobs,k_chunks,2] if resident else [8,k_chunks,jobs]
    for x in prepared:
        read_cta += ttnn.TensorAccessorArgs(x).get_compile_time_args()
    base = "experiments/sdpa-l2/bfp4-lofi-v2/" + ("resident/" if resident else "fullchip/")
    descriptor = ttnn.ProgramDescriptor(cbs=cbs,semaphores=[],kernels=[
        ttnn.KernelDescriptor(kernel_source=base+"reader.cpp", core_ranges=grid,
            compile_time_args=read_cta,runtime_args=read,
            defines=[] if resident else [("SDPA_READER_BARRIER_TILES","2")],config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=base+"writer.cpp",core_ranges=grid,
            compile_time_args=[jobs if resident else 8]+ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=write,config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=PREFIX+"compute.cpp",core_ranges=grid,
            compile_time_args=[jobs,k_chunks,struct.unpack("I",struct.pack("f",1/math.sqrt(128)))[0]],
            defines=list(defines.items()),config=ttnn.ComputeConfigDescriptor(
                math_fidelity=fidelity,fp32_dest_acc_en=False,dst_full_sync_en=False,math_approx_mode=True)),
    ])
    metadata = dict(defines=defines,fidelity=str(fidelity),fp32_dst=False,input_slots=2,
                    cb_specs=[(i,n,size,str(fmt)) for i,n,size,fmt in specs],cb_bytes=sum(n*s for _,n,s,_ in specs))
    return out, lambda: ttnn.generic_op(list(prepared)+[out],descriptor), metadata


def replay(device, invoke, out, expected, warmup, iters):
    invoke()  # Compile before trace capture.
    assert bits_equal(ttnn.to_torch(out), expected)
    trace = ttnn.begin_trace_capture(device,cq_id=0)
    invoke()
    ttnn.end_trace_capture(device,trace,cq_id=0)
    times = []
    try:
        # Mandatory even for --iters 0; capture alone is not a replay test.
        for _ in range(2):
            ttnn.execute_trace(device,trace,cq_id=0,blocking=True)
            assert bits_equal(ttnn.to_torch(out),expected), "Trace output changed"
        for i in range(warmup+iters):
            start = time.perf_counter()
            ttnn.execute_trace(device,trace,cq_id=0,blocking=True)
            if i >= warmup:
                times.append(1000*(time.perf_counter()-start))
        assert bits_equal(ttnn.to_torch(out),expected)
    finally:
        ttnn.release_trace(device,trace)
    return times


def run(device,args):
    candidate = (ROOT/args.candidate).resolve() if args.candidate else None
    if candidate:
        assert candidate.is_file() and candidate.is_relative_to(HERE), "Only isolated v3 compensated candidates allowed"
    before = source_hashes(candidate)
    resident = args.mode == "resident"
    n = 512 if resident else args.k_chunks*512
    assert args.q_length > 0 and args.q_length % 256 == 0
    assert not resident or args.q_length==256
    q,k,v = REPRO.make_inputs(1,args.q_length,n,128,args.seed,
                             "normal" if args.distribution == "zero_v" else args.distribution)
    if args.distribution == "zero_v":
        v.zero_()
    if getattr(args, "paired_maxima", False):
        assert not resident and not args.max_changing and args.k_chunks <= 16
        # Distinct V is retained. Identical adjacent K blocks force unchanged
        # maxima; powers-of-two between pairs force new positive maxima for
        # this normal-input stress without changing the numerical recipe.
        base_k = k[:, :, :512].clone()
        for block in range(args.k_chunks):
            k[:, :, 512*block:512*(block+1)] = (base_k.float() * (2.0 ** (block//2))).bfloat16()
    if args.max_changing:
        assert not resident
        # Force later K blocks to have progressively larger positive query
        # projections; distinct V unchanged. Original-input FP64 reference used.
        direction = q.float().mean(dim=2,keepdim=True)
        for b in range(args.k_chunks):
            k[:,:,512*b:512*(b+1)] = (k[:,:,512*b:512*(b+1)].float()+direction*(b+1)).bfloat16()
    originals = [q,k,v]
    host_hashes = [tensor_hash(x) for x in originals]
    inputs = [ttnn.from_torch(x,device=device,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT) for x in originals]
    device_hashes = [tensor_hash(ttnn.to_torch(x)) for x in inputs]
    assert device_hashes == host_hashes
    prepared = [CANONICAL.prepare(device,x,args.variant,is_q=i==0,cores=1) for i,x in enumerate(inputs)]
    represented = [ttnn.to_torch(x).float() for x in prepared]
    expected = [CANONICAL.PREP.MODEL.round_significand(q,7)]
    for x in (k,v):
        expected.append(CANONICAL.B4.host_rne_bfp4(x) if args.variant=="G" else
                        CANONICAL.PREP.MODEL.encode(x,"e5_b8","device").float())
    prep_mismatches = [int((a!=b).sum()) for a,b in zip(represented,expected)]
    assert prep_mismatches == [0,0,0], prep_mismatches
    prep_hashes = [tensor_hash(x) for x in represented]
    jobs = args.q_repeats if resident else args.q_length//256
    out,invoke,metadata = build(device,prepared,args.variant,resident=resident,jobs=jobs,k_chunks=args.k_chunks)
    invoke()
    baseline = ttnn.to_torch(out)
    assert torch.isfinite(baseline).all()
    canonical_equal = None
    if not resident:
        canonical_out = CANONICAL.attention(device,*prepared,args.variant,max_cores=1)
        canonical_equal = bits_equal(ttnn.to_torch(canonical_out),baseline)
        assert canonical_equal, "Adapter differs from canonical device_attention()"
    baseline_times = []
    actual, measured, candidate_metadata = baseline, [], metadata
    if candidate:
        cout,cinvoke,candidate_metadata = build(device,prepared,args.variant,resident=resident,jobs=jobs,
                                              k_chunks=args.k_chunks,candidate=candidate)
        cinvoke()
        actual = ttnn.to_torch(cout)
        # Numerical changes are evaluated against original-input FP64 below.
        # Replay determinism remains an exact raw-byte requirement.
        traces = []
        try:
            for operation in (invoke,cinvoke):
                trace = ttnn.begin_trace_capture(device,cq_id=0)
                operation()
                ttnn.end_trace_capture(device,trace,cq_id=0)
                traces.append(trace)
            for index, tensor in enumerate((out,cout)):
                for _ in range(2):
                    ttnn.execute_trace(device,traces[index],cq_id=0,blocking=True)
                    assert bits_equal(ttnn.to_torch(tensor), baseline if index == 0 else actual)
            for iteration in range(args.warmup+args.iters):
                order = (1,0) if (iteration+int(args.reverse)) % 2 else (0,1)
                for index in order:
                    start = time.perf_counter()
                    ttnn.execute_trace(device,traces[index],cq_id=0,blocking=True)
                    if iteration >= args.warmup:
                        (baseline_times if index==0 else measured).append(1000*(time.perf_counter()-start))
            assert bits_equal(ttnn.to_torch(out),baseline)
            assert bits_equal(ttnn.to_torch(cout),actual)
        finally:
            for trace in traces:
                ttnn.release_trace(device,trace)
    else:
        baseline_times = replay(device,invoke,out,baseline,args.warmup,args.iters)
        measured = baseline_times
    assert [tensor_hash(x) for x in originals] == host_hashes
    assert [tensor_hash(ttnn.to_torch(x)) for x in inputs] == device_hashes
    assert [tensor_hash(ttnn.to_torch(x).float()) for x in prepared] == prep_hashes
    assert source_hashes(candidate) == before, "Source changed during execution"
    reference = REPRO.reference(q,k,v)
    flops = 4*256*512*128*jobs*args.k_chunks
    def timing(times):
        median = statistics.median(times) if times else None
        return dict(replay_ms=times,median_ms=median,tflops_per_core=flops/(median*1e9) if median else None)
    return dict(arguments=vars(args),variant=args.variant,baseline_authority=str(BASELINE.relative_to(ROOT)),source_sha256=before,source_stable=True,
        baseline_metadata=metadata,candidate_metadata=candidate_metadata,
        original_input_sha256=host_hashes,prepared_input_sha256=prep_hashes,
        preprocessing_mismatches=prep_mismatches,original_host_immutable=True,original_device_immutable=True,
        prepared_device_immutable=True,canonical_adapter_equal=canonical_equal,
        baseline_candidate_equal=bits_equal(actual,baseline),
        numerical_comparison=NUMERICS.compare(baseline,actual,reference),
        output_equality_contract="raw uint8 including signed zero",preprocessing_equality_contract="exact decoded values; signed zeros equivalent",
        mandatory_trace_replays_per_kernel=2,trace_equal=True,output_sha256=tensor_hash(actual),
        baseline_output_sha256=tensor_hash(baseline),accuracy=NUMERICS.metrics(actual,reference),
        timing_order="alternating AB/BA" if candidate else "baseline only",
        useful_flops=flops,baseline=timing(baseline_times),candidate=timing(measured),
        warning="Repeated resident input favors unchanged maxima; compute scheduling only, preprocessing excluded"
        if resident else "Distinct-input attention with recurring DM; not no-DM throughput")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=("E","G"),required=True)
    parser.add_argument("--baseline",choices=("v1","v2"),default="v2")
    parser.add_argument("--time-distinct",action="store_true")
    parser.add_argument("--mode",choices=("resident","distinct"),default="resident")
    parser.add_argument("--k-chunks",type=int,default=512)
    parser.add_argument("--q-repeats",type=int,default=16)
    parser.add_argument("--q-length",type=int,default=256,help="Distinct path can execute multiple Q256 jobs on one core")
    parser.add_argument("--distribution",default="normal")
    parser.add_argument("--max-changing",action="store_true")
    parser.add_argument("--paired-maxima",action="store_true",
                        help="Distinct V; K block scales1,1,2,2,4,4,... test identity/fallback transitions")
    parser.add_argument("--candidate",help="Repo-relative private candidate header")
    parser.add_argument("--seed",type=int,default=1240)
    parser.add_argument("--device",type=int,default=0)
    parser.add_argument("--warmup",type=int,default=5)
    parser.add_argument("--iters",type=int,default=7)
    parser.add_argument("--reverse",action="store_true",help="Start paired measurements in BA order")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refusing to overwrite evidence"
    assert args.mode=="resident" or args.iters==0 or args.time_distinct
    global BASELINE
    BASELINE = (V1 / "combined_fence/compute_streaming.hpp" if args.baseline == "v1"
                else V2 / "identity_early/compute_streaming.hpp")
    torch.set_num_threads(8)
    device = ttnn.open_device(device_id=args.device,trace_region_size=8388608)
    try:
        result = run(device,args)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,default=str)+"\n")
        print(json.dumps(result,default=str),flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

