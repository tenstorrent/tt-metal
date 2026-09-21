# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Distinct Q/K/V, multiple Q jobs per core, unchanged canonical dataflow."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time

import torch
import ttnn

import bench_codegen as codegen
bench = codegen.bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--variant", choices=["C", "D"], required=True)
    parser.add_argument(
        "--candidate",
        choices=["O2", "O3", "Os"],
        required=True,
    )
    parser.add_argument("--q-length", type=int, default=2048)
    parser.add_argument("--k-length", type=int, default=8192)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1237)
    parser.add_argument("--distributions", default="normal,scaled_qk,outliers,common_k")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    output = bench.HERE / (args.label + ".json")
    assert not output.exists()
    spec = importlib.util.spec_from_file_location(
        "sprint_fullchip_adapter", bench.ROOT / "experiments/sdpa-l2/flux2-frontier-v1/device_attention.py"
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    flags = ["SDPA_SPRINT_IDENTITY4", "SDPA_SPRINT_STATE_UNPACK_BATCH", "SDPA_SPRINT_LAZY_INIT", "SDPA_SPRINT_MAX_SCAN"]
    if args.variant == "C":
        flags += ["SDPA_SPRINT_C_REFINE", "SDPA_SPRINT_C_REFINE_HOIST"]
    original_descriptor = ttnn.KernelDescriptor
    descriptors = []
    active_candidate = False

    def construct(*positional, **kwargs):
        if str(kwargs.get("kernel_source", "")).endswith("/compute.cpp"):
            kwargs["kernel_source"] = bench.SPRINT + "compute.cpp"
            kwargs["defines"] = list(kwargs["defines"]) + [(flag, "1") for flag in flags]
            if active_candidate:
                kwargs["kernel_source"] = "experiments/sdpa-l2/compute-sprint-v2/codegen/compute.cpp"
                kwargs["defines"].append(("SDPA_CODEGEN_OPT", {"O2": "2", "O3": "3", "Os": "4"}[args.candidate]))
        descriptors.append(
            {
                "source": kwargs["kernel_source"],
                "defines": kwargs.get("defines", []),
                "compile_time_args": kwargs.get("compile_time_args", []),
            }
        )
        return original_descriptor(*positional, **kwargs)

    records = []
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        for distribution in args.distributions.split(","):
            torch.manual_seed(args.seed)
            q = torch.randn(1, 1, args.q_length, 128)
            k = torch.randn(1, 1, args.k_length, 128)
            v = torch.randn_like(k)
            if distribution == "scaled_qk":
                q *= 2
                k *= 2
            elif distribution == "common_k":
                k += 32
            elif distribution == "outliers":
                q += (torch.rand_like(q) < 0.01) * torch.randn_like(q) * 10
                k += (torch.rand_like(k) < 0.01) * torch.randn_like(k) * 10
                v += (torch.rand_like(v) < 0.01) * torch.randn_like(v) * 10
            elif distribution == "changing_max":
                # Every Q row receives a strong positive feature; successive
                # K chunks increase it enough to force recurrent max updates.
                q[..., 0] = 8
                k[..., 0] = torch.arange(args.k_length) // 512 * 4
            else:
                assert distribution == "normal"
            inputs = [ttnn.from_torch(x.bfloat16(), device=device, layout=ttnn.TILE_LAYOUT) for x in (q, k, v)]
            oracle = None
            order = [
                candidate
                for round_index in range(args.rounds)
                for candidate in (("baseline", args.candidate) if round_index % 2 == 0 else (args.candidate, "baseline"))
            ]
            for order_index, candidate in enumerate(order):
                active_candidate = candidate != "baseline"
                descriptors.clear()
                ttnn.KernelDescriptor = construct
                try:

                    def invoke():
                        return adapter.attention(
                            device,
                            *inputs,
                            args.variant,
                            max_cores=args.cores,
                            q_chunk_size=256,
                            k_chunk_size=512,
                            reader_barrier_tiles=2,
                        )

                    out = invoke()
                    actual = ttnn.to_torch(out)
                    ttnn.deallocate(out)
                    if oracle is None:
                        oracle = actual.clone()
                    equal = bool(torch.equal(actual.view(torch.uint16), oracle.view(torch.uint16)))
                    timings = []
                    if args.iters:
                        trace = ttnn.begin_trace_capture(device, cq_id=0)
                        trace_out = invoke()
                        ttnn.end_trace_capture(device, trace, cq_id=0)
                        try:
                            for sample in range(5 + args.iters):
                                start = time.perf_counter()
                                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                                if sample >= 5:
                                    timings.append(1000 * (time.perf_counter() - start))
                            assert torch.equal(actual.view(torch.uint16), ttnn.to_torch(trace_out).view(torch.uint16))
                        finally:
                            ttnn.release_trace(device, trace)
                        ttnn.deallocate(trace_out)
                finally:
                    ttnn.KernelDescriptor = original_descriptor
                record = dict(
                    **vars(args),
                    current_distribution=distribution,
                    current_candidate=candidate,
                    round_index=order_index // 2,
                    full_output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
                    baseline_bitwise_equal=equal,
                    mismatches=int((actual.view(torch.uint16) != oracle.view(torch.uint16)).sum()),
                    median_ms=statistics.median(timings) if timings else None,
                    samples_ms=timings,
                    trace_equal=True if args.iters else None,
                    descriptors=list(descriptors),
                    source_sha256={
                        str(path.relative_to(bench.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in {bench.ROOT / d["source"] for d in descriptors}
                        | set(bench.HERE.glob("*.hpp"))
                        | {bench.HERE / "fullchip_codegen.py", bench.ROOT / (bench.SPRINT + "compute.cpp"), bench.ROOT / (bench.SPRINT + "compute_streaming.hpp"), bench.ROOT / (bench.SPRINT + "c_refine.hpp"), bench.ROOT / "experiments/sdpa-l2/flux2-frontier-v1/device_attention.py"}
                    },
                )
                records.append(record)
                torch.save(actual, bench.HERE / f"{args.label}-{distribution}-r{order_index // 2}-{candidate}.pt")
                output.write_text(json.dumps(records, indent=2) + "\n")
                print("RESULT " + json.dumps(record), flush=True)
                assert equal, f"Fullchip mismatch: {distribution} {candidate}"
            for tensor in inputs:
                ttnn.deallocate(tensor)
    finally:
        ttnn.KernelDescriptor = original_descriptor
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
