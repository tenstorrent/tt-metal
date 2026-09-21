# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Private function-optimization attributes around unchanged v1 C/D kernels."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location("v1_bench", ROOT / "experiments/sdpa-l2/compute-sprint-v1/fp32/bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.HERE = HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--variants", default="D,C")
    parser.add_argument("--levels", default="original,O3,O2,Os")
    parser.add_argument("--q-repeats", type=int, default=2)
    parser.add_argument("--k-chunks", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--seed", type=int, default=1236)
    opts = parser.parse_args()
    levels = opts.levels.split(",")
    assert set(levels) <= {"original", "O3", "O2", "Os"} and levels[0] == "original"
    target = HERE / (opts.label + ".json")
    assert not target.exists()
    original_descriptor = ttnn.KernelDescriptor
    original_to_torch = ttnn.to_torch
    records = []
    active_level = "original"
    descriptors = []
    host_outputs = []

    def read_output(*args, **kwargs):
        output = original_to_torch(*args, **kwargs)
        host_outputs.append(output.clone())
        return output

    def construct(*args, **kwargs):
        if str(kwargs.get("kernel_source", "")).endswith("/fp32/resident.cpp"):
            if active_level != "original":
                kwargs["kernel_source"] = "experiments/sdpa-l2/compute-sprint-v2/codegen/resident.cpp"
                kwargs["defines"] = list(kwargs["defines"]) + [("SDPA_CODEGEN_OPT", {"O2": "2", "O3": "3", "Os": "4"}[active_level])]
            descriptors.append(dict(source=kwargs["kernel_source"], defines=kwargs["defines"],
                                    compile_time_args=kwargs["compile_time_args"]))
        return original_descriptor(*args, **kwargs)

    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        distributions = [("normal", 512)] + [(d, 64) for d in ("scaled_qk", "outliers", "common_q", "common_k", "common_v", "constant_v", "uniform")] if opts.qualify else [("normal", opts.k_chunks)]
        for variant, distribution, chunks in [(v, d, n) for v in opts.variants.split(",") for d, n in distributions]:
            assert variant in ("C", "D")
            oracle = None
            for round_index in range(1 if opts.qualify else opts.rounds):
                for level in levels if round_index % 2 == 0 else list(reversed(levels)):
                    active_level = level
                    descriptors.clear()
                    host_outputs.clear()
                    args = argparse.Namespace(
                        label=f"{opts.label}-{variant}-{distribution}-r{round_index}-{level}", variant=variant,
                        candidate="state_lazy_scan" if variant == "D" else "c_refine_hoist_state_lazy_scan",
                        mode="accurate" if variant == "D" else "qk4_pv2", distribution=distribution, seed=opts.seed,
                        distinct_kv=opts.qualify, q_repeats=1 if opts.qualify else opts.q_repeats, k_chunks=chunks,
                        warmup=0 if opts.qualify else opts.warmup, iters=1 if opts.qualify else opts.iters, hybrid_block_pack=False, codegen_level=level)
                    ttnn.KernelDescriptor = construct
                    ttnn.to_torch = read_output
                    try:
                        record, actual = bench.run(device, args)
                    finally:
                        ttnn.KernelDescriptor = original_descriptor
                        ttnn.to_torch = original_to_torch
                    if oracle is None:
                        assert level == "original"
                        oracle = actual.clone()
                    record.update(round_index=round_index, actual_descriptors=list(descriptors),
                                  raw_bits_equal=bool(torch.equal(actual.view(torch.uint16), oracle.view(torch.uint16))),
                                  mismatch_count=int((actual.view(torch.uint16) != oracle.view(torch.uint16)).sum()),
                                  private_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                        for p in (HERE / "resident.cpp", Path(__file__))})
                    record["raw_trace_equal"] = bool(torch.equal(host_outputs[0].view(torch.uint16), host_outputs[-1].view(torch.uint16))) if args.iters else None
                    records.append(record)
                    target.write_text(json.dumps(records, indent=2) + "\n")
                    torch.save(actual, HERE / (args.label + ".pt"))
                    print("RESULT " + json.dumps(record), flush=True)
                    assert record["raw_bits_equal"], "Compiler experiment changed output bits"
                    if args.iters:
                        assert len(host_outputs) == 2 and record["raw_trace_equal"], "Trace replay changed output bits"
    finally:
        ttnn.KernelDescriptor = original_descriptor
        ttnn.to_torch = original_to_torch
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
