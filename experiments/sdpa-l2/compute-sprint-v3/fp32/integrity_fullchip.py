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

from pathlib import Path
spec = importlib.util.spec_from_file_location("fp32_v3_bench", Path(__file__).with_name("integrity_bench.py"))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
bench = driver.core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--variant", choices=["C", "D"], required=True)
    parser.add_argument(
        "--candidate",
        choices=["l1_inplace", "l1_both", "l1_early", "fusion_batch", "full_pv"],
        required=True,
    )
    parser.add_argument("--q-length", type=int, default=2048)
    parser.add_argument("--k-length", type=int, default=8192)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1237)
    parser.add_argument("--distributions", default="normal,scaled_qk,outliers,common_k")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--guard-stats", action="store_true")
    parser.add_argument("--baseline-kind", choices=["original", "late"], default="original")
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
            kwargs["kernel_source"] = "experiments/sdpa-l2/compute-sprint-v1/fp32/compute.cpp"
            if args.variant == "D":
                kwargs["kernel_source"] = "experiments/sdpa-l2/compute-sprint-v2/codegen/compute.cpp"
            kwargs["defines"] = list(kwargs["defines"]) + [(flag, "1") for flag in flags]
            if args.variant == "D" and not active_candidate:
                kwargs["defines"].append(("SDPA_CODEGEN_OPT", "2"))
            if active_candidate:
                kwargs["kernel_source"] = bench.SPRINT + ("compute_early.cpp" if args.candidate == "l1_early" else "compute.cpp")
                if args.variant == "D":
                    kwargs["defines"].append(("SDPA_V3_O2", "1"))
                flags_selected = {"l1_inplace": ["SDPA_V3_L1_INPLACE"], "l1_both": ["SDPA_V3_L1_INPLACE", "SDPA_V3_L1_BOTH"], "l1_early": ["SDPA_V3_L1_INPLACE", "SDPA_V3_L1_BOTH"], "full_pv": ["SDPA_V3_FULL_PV"],
                                  "fusion_batch": ["SDPA_V3_IDENTITY_FUSION", "SDPA_V3_FUSION_BATCH"]}[args.candidate]
                kwargs["defines"] += [(flag, "1") for flag in flags_selected]
                if args.guard_stats:
                    kwargs["defines"].append(("SDPA_V3_GUARD_STATS", "1"))
            if not active_candidate and args.baseline_kind == "late":
                kwargs["kernel_source"] = bench.SPRINT + "compute.cpp"
                kwargs["defines"] = [(key, value) for key, value in kwargs["defines"] if key != "SDPA_CODEGEN_OPT"]
                kwargs["defines"] += [("SDPA_V3_L1_INPLACE", "1"), ("SDPA_V3_L1_BOTH", "1")]
                if args.variant == "D":
                    kwargs["defines"].append(("SDPA_V3_O2", "1"))
        descriptors.append(
            {
                "source": kwargs["kernel_source"],
                "defines": kwargs.get("defines", []),
                "compile_time_args": kwargs.get("compile_time_args", []),
            }
        )
        return original_descriptor(*positional, **kwargs)

    source_paths = (
        list(bench.HERE.glob("*.hpp")) + list(bench.HERE.glob("*.cpp"))
        + list((bench.ROOT / "experiments/sdpa-l2/hybrid-mixed-v1").glob("**/*.hpp"))
        + list((bench.ROOT / "experiments/sdpa-l2/hybrid-mixed-v1").glob("**/*.h"))
        + list((bench.ROOT / "experiments/sdpa-l2/bfp4-lofi-v2").glob("*.hpp"))
        + list((bench.ROOT / "experiments/sdpa-l2/bfp4-lofi-v2/fullchip").glob("*.cpp"))
        + [bench.ROOT / "experiments/sdpa-l2/compute-sprint-v1/fp32/compute_streaming.hpp",
           bench.ROOT / "experiments/sdpa-l2/compute-sprint-v1/fp32/compute.cpp",
           bench.ROOT / "experiments/sdpa-l2/compute-sprint-v1/fp32/c_refine.hpp",
           bench.ROOT / "experiments/sdpa-l2/compute-sprint-v2/codegen/compute.cpp",
           Path(__file__).resolve(), Path(driver.__file__).resolve(), Path(bench.__file__).resolve(),
           Path(driver.numerics.__file__).resolve(), Path(adapter.__file__).resolve()]
        + [bench.ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py"]
    )
    def source_hashes():
        return {str(path.relative_to(bench.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}

    records = []
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        for distribution in args.distributions.split(","):
            repro_spec = importlib.util.spec_from_file_location("fp32_v3_repro", bench.ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")
            repro = importlib.util.module_from_spec(repro_spec)
            repro_spec.loader.exec_module(repro)
            torch.set_num_threads(8)
            q, k, v = bench.make_inputs(repro, args.q_length, args.k_length, args.seed, distribution)
            reference = repro.reference(q, k, v)
            inputs = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in (q, k, v)]
            host_input_hashes = [driver.digest(x) for x in (q, k, v)]
            device_input_hashes_before = [driver.digest(ttnn.to_torch(x)) for x in inputs]
            assert device_input_hashes_before == host_input_hashes, "Device conversion changed BF16 input bits"
            oracle = None
            order = [
                candidate
                for round_index in range(args.rounds)
                for candidate in (("baseline", args.candidate) if round_index % 2 == 0 else (args.candidate, "baseline"))
            ]
            for order_index, candidate in enumerate(order):
                active_candidate = candidate != "baseline"
                descriptors.clear()
                source_hashes_before = source_hashes()
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
                    if True:
                        trace = ttnn.begin_trace_capture(device, cq_id=0)
                        trace_out = invoke()
                        ttnn.end_trace_capture(device, trace, cq_id=0)
                        try:
                            replay_count = max(2, 5 + args.iters if args.iters else 2)
                            for sample in range(replay_count):
                                start = time.perf_counter()
                                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                                elapsed_ms = 1000 * (time.perf_counter() - start)
                                if args.iters and sample >= 5:
                                    timings.append(elapsed_ms)
                                if sample < 2:
                                    assert torch.equal(actual.view(torch.uint16), ttnn.to_torch(trace_out).view(torch.uint16)), "Replay bits differ"
                            assert torch.equal(actual.view(torch.uint16), ttnn.to_torch(trace_out).view(torch.uint16))
                        finally:
                            ttnn.release_trace(device, trace)
                        ttnn.deallocate(trace_out)
                finally:
                    ttnn.KernelDescriptor = original_descriptor
                device_input_hashes_after = [driver.digest(ttnn.to_torch(x)) for x in inputs]
                assert device_input_hashes_after == device_input_hashes_before, "Kernel modified input DRAM"
                source_hashes_after = source_hashes()
                assert source_hashes_after == source_hashes_before, "Source closure changed during run"
                record = dict(
                    device_input_hashes_before=device_input_hashes_before,
                    device_input_hashes_after=device_input_hashes_after, host_input_hashes=host_input_hashes,
                    source_hashes_before=source_hashes_before, source_hashes_after=source_hashes_after,
                    device_inputs_immutable=True, source_closure_immutable=True, actual_replays=replay_count,
                    **vars(args),
                    current_distribution=distribution,
                    current_candidate=candidate,
                    round_index=order_index // 2,
                    full_output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
                    baseline_bitwise_equal=equal,
                    mismatches=int((actual.view(torch.uint16) != oracle.view(torch.uint16)).sum()),
                    median_ms=statistics.median(timings) if timings else None,
                    samples_ms=timings,
                    trace_equal=True,
                    descriptors=list(descriptors),
                    source_sha256={
                        str(path.relative_to(bench.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in {bench.ROOT / d["source"] for d in descriptors}
                        | set(bench.HERE.glob("*.hpp"))
                        | {bench.ROOT / "experiments/sdpa-l2/compute-sprint-v1/fp32/compute_streaming.hpp",
                           bench.ROOT / "experiments/sdpa-l2/compute-sprint-v1/fp32/compute.cpp",
                           bench.ROOT / "experiments/sdpa-l2/compute-sprint-v2/codegen/compute.cpp"}
                        | {Path(__file__).resolve(), bench.HERE / "core.py", bench.HERE / "bench.py", bench.ROOT / (bench.SPRINT + "compute.cpp"), bench.ROOT / (bench.SPRINT + "compute_streaming.hpp"), bench.ROOT / "experiments/sdpa-l2/compute-sprint-v1/fp32/c_refine.hpp", bench.ROOT / "experiments/sdpa-l2/flux2-frontier-v1/device_attention.py"}
                    },
                )
                record.update(driver.accuracy(actual, reference, oracle))
                record.update(driver.numerics.compare(oracle, actual, reference))
                record["raw_trace_equal"] = True
                record["input_hashes"] = [driver.digest(x) for x in (q, k, v)]
                record["useful_flops"] = 4 * args.q_length * args.k_length * 128
                record["tflops_per_core"] = record["useful_flops"] / (record["median_ms"] * 1e9 * args.cores) if record["median_ms"] else None
                record["shared_metrics_sha256"] = hashlib.sha256(Path(driver.numerics.__file__).read_bytes()).hexdigest()
                records.append(record)
                torch.save(actual, bench.HERE / f"{args.label}-{distribution}-r{order_index // 2}-{candidate}.pt")
                output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
                print("RESULT " + json.dumps(record, allow_nan=False), flush=True)
                # Numerical gate rejections are data, not device faults.
            for tensor in inputs:
                ttnn.deallocate(tensor)
    finally:
        ttnn.KernelDescriptor = original_descriptor
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
