"""B group2 paired timings against both frozen general and early-guard controls."""

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch
import ttnn

import qualify as Q

OLD = Q.load("v3_review_frozen_builder", "experiments/sdpa-l2/compute-sprint-v2/review/bench.py")


def run(args):
    torch.set_num_threads(8)
    target = Q.HERE / (args.label + ".json")
    assert not target.exists()
    input_args = SimpleNamespace(q_length=256, k_length=512*args.k_chunks if args.distinct_kv else 512,
                                 seed=args.seed)
    q, k, v = Q.make_inputs(input_args, args.distribution)
    reference = Q.REPRO.reference(q, k, v)
    # Repeating identical KV does not change the exact normalized attention.
    pins = Q.source_pins()
    input_hashes = [Q.digest(x) for x in (q, k, v)]
    variants = ["v1", "v2_early", "group2"]
    original = ttnn.KernelDescriptor
    traces, outputs, records, tensors = {}, {}, {}, []
    mode = "v1"
    def descriptor(*pos, **kwargs):
        if mode == "group2" and kwargs["kernel_source"].endswith("/resident.cpp"):
            kwargs["kernel_source"] = "experiments/sdpa-l2/compute-sprint-v3/review/resident.cpp"
        return original(*pos, **kwargs)
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        tensors = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in (q,k,v)]
        for mode in variants:
            ttnn.KernelDescriptor = descriptor
            try:
                out, invoke, provenance = OLD.build(device, args, tensors,
                    {"v1":"canonical", "v2_early":"identity_early", "group2":"disabled"}[mode])
                invoke()
                actual = ttnn.to_torch(out)
                outputs[mode] = actual.clone()
                trace = ttnn.begin_trace_capture(device, cq_id=0)
                invoke()
                ttnn.end_trace_capture(device, trace, cq_id=0)
                traces[mode] = (trace, out)
            finally:
                ttnn.KernelDescriptor = original
            records[mode] = dict(provenance=provenance, metrics=Q.NUMERICS.metrics(actual, reference),
                                 output_sha256=Q.digest(actual), times_ms=[])
        assert torch.equal(outputs["v1"].view(torch.uint16), outputs["v2_early"].view(torch.uint16))
        gates = {name: Q.NUMERICS.compare(outputs[name], outputs["group2"], reference)
                 for name in ("v1", "v2_early")}
        # Screening numerical rejects are still recorded and close normally.
        for index in range(args.warmup + args.iters):
            order = variants if index % 2 == 0 else list(reversed(variants))
            for mode in order:
                trace, out = traces[mode]
                start = time.perf_counter()
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                elapsed = 1000*(time.perf_counter()-start)
                if index >= args.warmup:
                    records[mode]["times_ms"].append(elapsed)
        useful = 4*256*512*128*args.q_repeats*args.k_chunks
        for mode, (trace, out) in traces.items():
            actual = ttnn.to_torch(out)
            replay_checks = []
            for _ in range(2):
                replay_checks.append(torch.equal(actual.view(torch.uint16), outputs[mode].view(torch.uint16)))
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                actual = ttnn.to_torch(out)
            replay_checks.append(torch.equal(actual.view(torch.uint16), outputs[mode].view(torch.uint16)))
            records[mode]["trace_raw_equal"] = replay_checks
            assert all(replay_checks), "Eager/trace raw bits differ"
            median = statistics.median(records[mode]["times_ms"])
            records[mode].update(median_ms=median, tflops_per_core=useful/(median*1e9),
                                raw_equal_v1=torch.equal(outputs[mode].view(torch.uint16),
                                                         outputs["v1"].view(torch.uint16)))
        for tensor, expected in zip(tensors, input_hashes):
            assert Q.digest(ttnn.to_torch(tensor)) == expected
        assert pins == Q.source_pins()
        report = dict(arguments=vars(args), results=records, comparisons=gates,
                      all_numerical_gates_pass=all(x["acceptance"]["pass"] for x in gates.values()),
                      useful_flops=useful, input_sha256=input_hashes, source_sha256=pins,
                      input_immutability_pass=True, source_immutability_pass=True,
                      note=("Repeated same Q job on distinct KV including recurring DM" if args.distinct_kv
                            else "Resident repeated identical Q/K/V; no recurring input DM; not a model speedup"))
        target.write_text(json.dumps(report,indent=2)+"\n")
        print("RESULT "+json.dumps(report),flush=True)
    finally:
        ttnn.KernelDescriptor = original
        for trace, _ in traces.values():
            ttnn.release_trace(device,trace)
        ttnn.close_device(device)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--label",required=True)
    p.add_argument("--variant",default="B",choices=("B",))
    p.add_argument("--distribution",default="normal")
    p.add_argument("--seed",type=int,default=20260918)
    p.add_argument("--distinct-kv",action="store_true")
    p.add_argument("--q-repeats",type=int,default=16)
    p.add_argument("--k-chunks",type=int,default=512)
    p.add_argument("--warmup",type=int,default=12)
    p.add_argument("--iters",type=int,default=10)
    run(p.parse_args())
