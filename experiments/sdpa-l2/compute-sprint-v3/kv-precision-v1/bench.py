"""Controlled E-family KV storage ablation; frozen compute and dataflow sources."""
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
V3 = HERE.parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location("kv_original_collect", V3 / "pareto/collect.py")
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)
LOW, CANON, REPRO, NUM = OLD.LOW, OLD.CANON, OLD.REPRO, OLD.NUM
FORMATS = ("E_bf16", "E_bfp8", "E_bfp4")


def pins():
    result = OLD.pins()
    result[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def prepare(device, originals, host_inputs, variant):
    prepared, expected = [], []
    for index, x in enumerate(originals):
        if variant == "E_bf16":
            out, invoke, _ = CANON.PREP.build(device, x, bits=7 if index == 0 else 5,
                                             output_format="bf16", ncores=1)
            invoke()
        else:
            out = CANON.prepare(device, x, "E" if variant == "E_bfp8" else "G", is_q=index == 0, cores=1)
        prepared.append(out)
        h = host_inputs[index]
        if index == 0 or variant == "E_bf16":
            expected.append(CANON.PREP.MODEL.round_significand(h, 7 if index == 0 else 5))
        elif variant == "E_bfp8":
            expected.append(CANON.PREP.MODEL.encode(h, "e5_b8", "device").float())
        else:
            expected.append(CANON.B4.host_rne_bfp4(h))
    actual = [ttnn.to_torch(x).float() for x in prepared]
    assert all(torch.equal(a, b.float()) for a, b in zip(actual, expected)), "Preparation mismatch"
    return prepared, [OLD.digest(x) for x in actual]


def build(device, prepared, variant, *, resident, jobs, k_chunks):
    # Same program as compensated/benchmark.py::build. Only KV format/page size varies.
    assert variant in FORMATS and jobs > 0 and k_chunks > 0
    fp32, fast, defines, fidelity = CANON.recipe("E")
    assert not fp32 and fast and fidelity == ttnn.MathFidelity.LoFi
    defines = dict(defines)
    defines["SDPA_K_CHUNK_TILES"] = "16"
    defines["SDPA_SPRINT_CANDIDATE_HEADER"] = '"' + str(OLD.GROUPED.relative_to(ROOT)) + '"'
    kv_type, kv_bytes = {"E_bf16": (ttnn.bfloat16, 2048), "E_bfp8": (ttnn.bfloat8_b, 1088),
                         "E_bfp4": (ttnn.bfloat4_b, 576)}[variant]
    assert [x.get_dtype() for x in prepared] == [ttnn.bfloat16, kv_type, kv_type]
    assert tuple(prepared[0].shape) == (1, 1, 256 if resident else 256 * jobs, 128)
    assert tuple(prepared[1].shape) == (1, 1, 512 if resident else 512 * k_chunks, 128)
    assert tuple(prepared[2].shape) == tuple(prepared[1].shape)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    bf = ttnn.bfloat16
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
        ttnn.KernelDescriptor(kernel_source=LOW.PREFIX+"compute.cpp",core_ranges=grid,
            compile_time_args=[jobs,k_chunks,struct.unpack("I",struct.pack("f",1/math.sqrt(128)))[0]],
            defines=list(defines.items()),config=ttnn.ComputeConfigDescriptor(
                math_fidelity=fidelity,fp32_dest_acc_en=False,dst_full_sync_en=False,math_approx_mode=True)),
    ])
    metadata = dict(defines=defines,fidelity=str(fidelity),fp32_dst=False,input_slots=2,
                    cb_specs=[(i,n,size,str(fmt)) for i,n,size,fmt in specs],cb_bytes=sum(n*s for _,n,s,_ in specs),
                    kv_tile_bytes=kv_bytes, compute_source=LOW.PREFIX+"compute.cpp")
    return out, lambda: ttnn.generic_op(list(prepared)+[out],descriptor), metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "accuracy", "perf"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(8)
    old_data = json.loads((V3 / "pareto/matched-v1.json").read_text())
    old_rows = {(r["variant"], r["suite"], r["distribution"], r["k_length"]): r for r in old_data["records"]}
    if args.mode == "accuracy":
        cases = [("core", 256, n, d) for n in (4096,32768,262144)
                 for d in ("normal","clipped","scaled_down","scaled_qk","outliers","uniform")]
        cases += [("stress",256,32768,d) for d in ("common_q","common_k","common_v")]
    elif args.mode == "smoke":
        cases = [("smoke",512,n,d) for n,d in ((512,"normal"),(1024,"uniform"),
                                              (1536,"normal"),(1536,"constant_v"),(1536,"zero_v"))]
    else:
        cases = [("resident",256,512,"normal")]
    report = dict(mode=args.mode, seed=20260919, records=[], source_sha256=pins(), complete=False,
                  contract="LoFi, BF16 DST, frozen group2_valid state/exp, Q256/K512/D128, unchanged tile counts and two input slots.")
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        for suite, qlen, n, dist in cases:
            host = REPRO.make_inputs(1, qlen, n, 128, report["seed"],
                     "normal" if dist in ("clipped","scaled_down","zero_v") else dist)
            if dist == "clipped": host = [x.clamp(-2,2) for x in host]
            if dist == "scaled_down": host[:2] = [(x.float()*.25).bfloat16() for x in host[:2]]
            if dist == "zero_v": host[2].zero_()
            hashes = [OLD.digest(x) for x in host]
            reference = REPRO.reference(*host)
            originals = [ttnn.from_torch(x,device=device,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT) for x in host]
            assert [OLD.digest(ttnn.to_torch(x)) for x in originals] == hashes
            active = {}
            for variant in FORMATS:
                prepared, prep_hashes = prepare(device, originals, host, variant)
                resident = args.mode == "perf"
                jobs, chunks = (16,512) if resident else (qlen//256,n//512)
                out, invoke, metadata = build(device, prepared, variant, resident=resident,jobs=jobs,k_chunks=chunks)
                invoke()
                actual = ttnn.to_torch(out)
                assert torch.isfinite(actual).all()
                legacy_equal = None
                if variant != "E_bf16" and args.mode == "smoke":
                    legacy, legacy_invoke, legacy_metadata = LOW.build(device, prepared,
                        "E" if variant == "E_bfp8" else "G", resident=resident,jobs=jobs,k_chunks=chunks,candidate=OLD.GROUPED)
                    legacy_invoke()
                    assert OLD.digest(ttnn.to_torch(legacy)) == OLD.digest(actual)
                    assert all(metadata[k] == v for k,v in legacy_metadata.items())
                    ttnn.deallocate(legacy)
                    legacy_equal = True
                if variant != "E_bf16" and args.mode == "accuracy":
                    previous = old_rows[("E" if variant == "E_bfp8" else "G",suite,dist,n)]
                    assert previous["input_sha256"] == hashes
                    assert previous["output_sha256"] == OLD.digest(actual), "Old E/G output changed"
                    legacy_equal = True
                trace = ttnn.begin_trace_capture(device,cq_id=0)
                invoke()
                ttnn.end_trace_capture(device,trace,cq_id=0)
                for _ in range(2):
                    ttnn.execute_trace(device,trace,cq_id=0,blocking=True)
                    assert OLD.digest(ttnn.to_torch(out)) == OLD.digest(actual)
                row = dict(variant=variant,suite=suite,distribution=dist,k_length=n,query_rows=qlen,
                           metrics=NUM.metrics(actual,reference),input_sha256=hashes,prepared_sha256=prep_hashes,
                           output_sha256=OLD.digest(actual),raw_trace_equal=True,actual_replays=2,
                           preprocessing_exact=True,legacy_equal=legacy_equal,metadata=metadata,samples_ms=[])
                active[variant] = (trace,out,prepared,row)
            # Assert only CB1/2 storage formats and bytes differ between family members.
            first = active[FORMATS[0]][3]["metadata"]
            for variant in FORMATS[1:]:
                m = active[variant][3]["metadata"]
                for key in ("defines","fidelity","fp32_dst","input_slots","compute_source"):
                    assert m[key] == first[key]
                assert [s for s in m["cb_specs"] if s[0] not in (1,2)] == [s for s in first["cb_specs"] if s[0] not in (1,2)]
            if args.mode == "perf":
                for iteration in range(21):
                    order = FORMATS[iteration%3:] + FORMATS[:iteration%3]
                    if iteration%2: order = order[::-1]
                    for variant in order:
                        trace,out,prepared,row = active[variant]
                        start = time.perf_counter()
                        ttnn.execute_trace(device,trace,cq_id=0,blocking=True)
                        elapsed = 1000*(time.perf_counter()-start)
                        if iteration >= 9: row["samples_ms"].append(elapsed)
            for variant,(trace,out,prepared,row) in active.items():
                assert OLD.digest(ttnn.to_torch(out)) == row["output_sha256"]
                assert [OLD.digest(ttnn.to_torch(x).float()) for x in prepared] == row["prepared_sha256"]
                assert [OLD.digest(ttnn.to_torch(x)) for x in originals] == hashes
                row.update(inputs_immutable=True, prepared_immutable=True)
                if row["samples_ms"]:
                    flops = 4*256*512*128*16*512
                    median = statistics.median(row["samples_ms"])
                    row.update(median_ms=median,tflops_per_core=flops/(median*1e9),useful_flops=flops,
                               q_repeats=16,k_chunks=512,actual_replays=23)
                report["records"].append(row)
                ttnn.release_trace(device,trace)
                ttnn.deallocate(out)
                for x in prepared: ttnn.deallocate(x)
                print(variant,suite,n,dist,row["metrics"]["l2_pct"],row.get("tflops_per_core"),flush=True)
            for x in originals: ttnn.deallocate(x)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        assert pins() == report["source_sha256"]
        report.update(complete=True,selected_sources_immutable=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
