"""No external special rounding; frozen E compute and normal device typecasts."""
import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
spec = importlib.util.spec_from_file_location("no_round_kv", V3 / "kv-precision-v1/bench.py")
KV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(KV)
OLD = KV.OLD
FORMATS = KV.FORMATS


def pins():
    p = KV.pins()
    p[str(Path(__file__).relative_to(KV.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "accuracy", "perf"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(8)
    baseline = json.loads((V3 / "kv-precision-v1/accuracy-v1.json").read_text())
    assert baseline["complete"]
    prior = {(r["variant"],r["suite"],r["k_length"],r["distribution"]):r for r in baseline["records"]}
    if args.mode == "accuracy":
        cases = [("core",256,n,d) for n in (4096,32768,262144)
                 for d in ("normal","clipped","scaled_down","scaled_qk","outliers","uniform")]
        cases += [("stress",256,32768,d) for d in ("common_q","common_k","common_v")]
    elif args.mode == "smoke":
        cases = [("smoke",512,n,d) for n,d in ((512,"normal"),(1024,"uniform"),
                  (1536,"normal"),(1536,"constant_v"),(1536,"zero_v"))]
    else:
        cases = [("resident",256,512,"normal")]
    report = dict(mode=args.mode,seed=20260919,records=[],complete=False,source_sha256=pins(),
                  preprocessing="Q unchanged BF16; KV unchanged BF16 or ttnn.typecast to BFP8/BFP4; no custom rounding",
                  numerical_contract="Frozen E-family kernel, including internal arithmetic/rounding; Q256/K512/D128, two input slots")
    device = ttnn.open_device(device_id=0,trace_region_size=16777216)
    try:
        for suite,qlen,n,dist in cases:
            host = KV.REPRO.make_inputs(1,qlen,n,128,report["seed"],
                     "normal" if dist in ("clipped","scaled_down","zero_v") else dist)
            if dist == "clipped": host = [x.clamp(-2,2) for x in host]
            if dist == "scaled_down": host[:2] = [(x.float()*.25).bfloat16() for x in host[:2]]
            if dist == "zero_v": host[2].zero_()
            hashes = [OLD.digest(x) for x in host]
            reference = KV.REPRO.reference(*host)
            originals = [ttnn.from_torch(x,device=device,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT) for x in host]
            assert [OLD.digest(ttnn.to_torch(x)) for x in originals] == hashes
            active = {}
            for variant in FORMATS:
                if variant == "E_bf16":
                    prepared = list(originals)
                else:
                    dtype = ttnn.bfloat8_b if variant == "E_bfp8" else ttnn.bfloat4_b
                    prepared = [originals[0]] + [ttnn.typecast(x,dtype) for x in originals[1:]]
                represented = [ttnn.to_torch(x).float() for x in prepared]
                assert torch.equal(represented[0],host[0].float())
                assert all(torch.isfinite(x).all() for x in represented)
                if variant == "E_bf16":
                    assert all(torch.equal(x,y.float()) for x,y in zip(represented,host))
                before = [OLD.digest(x) for x in represented]
                resident = args.mode == "perf"
                jobs,chunks = (16,512) if resident else (qlen//256,n//512)
                out,invoke,metadata = KV.build(device,prepared,variant,resident=resident,jobs=jobs,k_chunks=chunks)
                invoke()
                actual = ttnn.to_torch(out)
                assert torch.isfinite(actual).all()
                previous = prior.get((variant,suite,n,dist))
                if previous:
                    assert previous["input_sha256"] == hashes
                    assert previous["metadata"] == json.loads(json.dumps(metadata))
                trace = ttnn.begin_trace_capture(device,cq_id=0)
                invoke()
                ttnn.end_trace_capture(device,trace,cq_id=0)
                for _ in range(2):
                    ttnn.execute_trace(device,trace,cq_id=0,blocking=True)
                    assert OLD.digest(ttnn.to_torch(out)) == OLD.digest(actual)
                row = dict(variant=variant+"_plain",base_variant=variant,suite=suite,distribution=dist,
                    k_length=n,query_rows=qlen,metrics=KV.NUM.metrics(actual,reference),
                    baseline_metrics=previous["metrics"] if previous else None,input_sha256=hashes,
                    prepared_sha256=before,output_sha256=OLD.digest(actual),metadata=metadata,
                    raw_trace_equal=True,actual_replays=2,samples_ms=[])
                active[variant] = (trace,out,prepared,row)
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
                row.update(inputs_immutable=True,prepared_immutable=True)
                if row["samples_ms"]:
                    flops = 4*256*512*128*16*512
                    median = statistics.median(row["samples_ms"])
                    row.update(median_ms=median,tflops_per_core=flops/(median*1e9),useful_flops=flops,
                               q_repeats=16,k_chunks=512,actual_replays=23)
                report["records"].append(row)
                ttnn.release_trace(device,trace)
                ttnn.deallocate(out)
                if variant != "E_bf16":
                    for x in prepared[1:]: ttnn.deallocate(x)
                print(row["variant"],suite,n,dist,row["metrics"]["l2_pct"],row.get("tflops_per_core"),flush=True)
            for x in originals: ttnn.deallocate(x)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        assert pins() == report["source_sha256"]
        report.update(complete=True,selected_sources_immutable=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
