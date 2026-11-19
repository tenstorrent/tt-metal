# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated decode-reduction candidate correctness and traced host timing.

No model weights. Default BF16 is a candidate, not an accepted production change.
--accumulation fp32 is precision-changing diagnostic evidence only. Every layout
conversion and copy to a persistent destination is included in each trace.

Serialize hardware externally; clear watcher/profiler/tracker for timing::

 TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto \
 timeout 240 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_moe_reduction_candidate \
 --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/moe_reduction_candidate_bf16_v1.json

Use --tracked-correctness with TT_METAL_TRACE_ALLOC_TRACKING=1 and
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 for correctness only (no timing).
--cpu-only validates fixtures without importing TTNN. Default real fixtures are
layer0/token42+43/batch from moe_decode_capture_v1.json; --fixture may override.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch

from .probe_batch_reduction_counterfactual import batched_reduction
from .probe_moe_reduction_rows import ROOT, compare, make_reducer, synthetic_fixture, validate_fixture


def load_fixtures(args):
    fixtures = [
        ("synthetic_" + pattern, synthetic_fixture(pattern, 162 + index))
        for index, pattern in enumerate(("zero_local_rank0", "one_local_rank0", "multiple_each_rank"))
    ]
    if args.fixture:
        paths = args.fixture
    else:
        manifest = json.loads((ROOT / "doc/full_model/moe_decode_capture_v1.json").read_text())
        entries = [
            item
            for item in manifest["fixtures"]
            if item["layer"] == 0 and item["token"] in (42, 43) and item["geometry"] == "batch"
        ]
        assert sorted(item["token"] for item in entries) == [42, 43], "Need both real layer0 decode fixtures"
        paths = [Path(item["path"]) for item in sorted(entries, key=lambda item: item["token"])]
        for entry, path in zip(sorted(entries, key=lambda item: item["token"]), paths):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
    for path in paths:
        fixture = torch.load(path, weights_only=True, map_location="cpu")
        validate_fixture(fixture)
        fixture = dict(fixture)
        fixture["fixture_path"] = str(path)
        fixture["fixture_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        fixtures.append((path.stem, fixture))
    return fixtures


def preflight(args):
    assert "TT_METAL_WATCHER" not in os.environ, "Watcher must be absent, including value0"
    assert not os.environ.get("TT_METAL_DEVICE_PROFILER", "").startswith("1")
    assert not os.environ.get("TT_METAL_DEVICE_PROFILER_STREAMING", "").startswith("1")
    assert os.environ.get("TT_METAL_STREAMING_PROFILER") in (None, "", "0")
    assert os.environ.get("TT_METAL_TRACE_PROFILER") in (None, "", "0")
    assert int(os.environ.get("TT_METAL_PROFILE_PERF_COUNTERS") or "0") == 0
    tracking = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0")
    if args.tracked_correctness:
        assert tracking == "1", "Enable tracking before imports"
        assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    else:
        assert tracking in ("", "0"), "Clean timing requires allocation tracking disabled"
        assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACEBACKS") in (
            None,
            "",
            "0",
        ), "Disable allocation diagnostics for timing"
    assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "Restore accepted mesh descriptor"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, action="append")
    parser.add_argument("--accumulation", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--tracked-correctness", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.iterations <= 100 and 1 <= args.rounds <= 5 and 1 <= args.warmup <= 5
    assert not args.output.exists(), "Choose a new diagnostic output path"
    torch.set_num_threads(4)
    report = {
        "completed": False,
        "acceptance_eligible": False,
        "scope": "isolated reduction candidate; full-model correctness and performance still required",
        "accumulation": args.accumulation,
        "precision_changed": args.accumulation == "fp32",
        "tracked_correctness": args.tracked_correctness,
        "cpu_only": args.cpu_only,
        "timing_includes": ["ownership mask", "layout conversions", "reduction", "persistent-output copy"],
        "timing_excludes": ["host input upload", "output readback", "trace capture", "compilation"],
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_DEVICE_PROFILER_STREAMING",
                "TT_METAL_STREAMING_PROFILER",
                "TT_METAL_TRACE_PROFILER",
                "TT_METAL_TRACE_ALLOC_TRACEBACKS",
                "TT_METAL_PROFILE_PERF_COUNTERS",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_MESH_GRAPH_DESC_PATH",
            )
        },
    }
    mesh, traces, started = None, [], time.monotonic()
    try:
        fixtures = load_fixtures(args)
        for _, fixture in fixtures:
            validate_fixture(fixture)
        report["fixtures"] = [
            {
                "name": name,
                "provenance": fixture.get("provenance", {}),
                "path": fixture.get("fixture_path"),
                "sha256": fixture.get("fixture_sha256"),
            }
            for name, fixture in fixtures
        ]
        if args.cpu_only:
            report["completed"] = True
            return
        preflight(args)
        from ttnn.tools import trace_allocation_tracker

        import ttnn as tt

        if not args.tracked_correctness:
            assert not trace_allocation_tracker.TRACE_ALLOC_TRACKING, "Separate allocation tracking from timing"
            assert not trace_allocation_tracker.TRACE_ALLOC_DIAGNOSTICS, "Disable allocation diagnostics for timing"
        tt.set_fabric_config(tt.FabricConfig.FABRIC_1D_RING)
        mesh = tt.open_mesh_device(tt.MeshShape(1, 4), l1_small_size=32768, trace_region_size=32 * 1024 * 1024)
        moe = make_reducer(tt, mesh)
        compute = (
            moe.compute
            if args.accumulation == "fp32"
            else tt.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=tt.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
                dst_full_sync_en=False,
            )
        )

        def upload(fixture, rows):
            n = len(rows)
            values = fixture["combined"][:, rows].reshape(4, 1, n, 10, 2560)
            combined = tt.from_torch(
                values.contiguous(),
                device=mesh,
                dtype=tt.bfloat16,
                layout=tt.ROW_MAJOR_LAYOUT,
                mesh_mapper=tt.ShardTensorToMesh(mesh, dim=0),
            )
            weights = tt.from_torch(
                fixture["weights"][rows].reshape(1, 1, n, 10).contiguous(),
                device=mesh,
                dtype=tt.bfloat16,
                layout=tt.ROW_MAJOR_LAYOUT,
                mesh_mapper=tt.ReplicateTensorToMesh(mesh),
            )
            ids = tt.from_torch(
                fixture["ids"].int()[rows].reshape(1, 1, n, 10).contiguous(),
                device=mesh,
                dtype=tt.uint16,
                layout=tt.ROW_MAJOR_LAYOUT,
                mesh_mapper=tt.ReplicateTensorToMesh(mesh),
            )
            return combined, weights, ids

        def read(value, n):
            return torch.stack([tt.to_torch(rank).reshape(n, 2560).clone() for rank in tt.get_device_tensors(value)])

        def invoke(strategy, inputs, n):
            return moe._reduce(*inputs, n) if strategy == "native" else batched_reduction(tt, moe, *inputs, n, compute)

        jobs = []
        report["correctness"] = []
        for name, fixture in fixtures:
            print("REDUCTION_COMPONENT_PHASE", "eager_correctness", name, flush=True)
            single = torch.cat([read(moe._reduce(*upload(fixture, [row]), 1), 1) for row in (0, 1)], dim=1)
            shapes = {
                "b2": [0, 1],
                "b2_duplicate": [0, 0],
                "b2_swap": [1, 0],
                "b32": [0, 1] * 16,
                "b32_duplicate": [0] * 32,
                "b32_swap": [1, 0] * 16,
            }
            outputs = {}
            item = {"fixture": name, "comparisons_to_single": {}, "invariants": {}}
            for shape, rows in shapes.items():
                inputs = upload(fixture, rows)
                n = len(rows)
                outputs[shape] = {}
                for strategy in ("native", "candidate"):
                    output = read(invoke(strategy, inputs, n), n)
                    repeated = read(invoke(strategy, inputs, n), n)
                    outputs[shape][strategy] = output
                    item["comparisons_to_single"][shape + "/" + strategy] = compare(single[:, rows], output)
                    item["invariants"][shape + "/" + strategy + "/repeat"] = compare(output, repeated)
                    if shape in ("b2", "b32"):
                        destination = tt.empty(
                            (1, 1, n, 2560),
                            dtype=tt.bfloat16,
                            layout=tt.TILE_LAYOUT,
                            device=mesh,
                            memory_config=tt.DRAM_MEMORY_CONFIG,
                        )
                        jobs.append(
                            {
                                "fixture": name,
                                "n": n,
                                "strategy": strategy,
                                "inputs": inputs,
                                "destination": destination,
                                "expected": output,
                            }
                        )
            for strategy in ("native", "candidate"):
                base = outputs["b2"][strategy]
                for shape, rows in shapes.items():
                    item["invariants"][shape + "/" + strategy + "/row_mapping"] = compare(
                        base[:, rows], outputs[shape][strategy]
                    )
            report["correctness"].append(item)
            assert all(value["exact"] for value in item["invariants"].values()), "Eager same-path invariance failed"

        def body(job):
            tt.copy(invoke(job["strategy"], job["inputs"], job["n"]), job["destination"])

        # All shapes, immutable inputs and persistent destinations exist now.
        # Warm every exact body before capturing any trace. No eager device
        # operations or new device tensors are created after the first capture.
        print("REDUCTION_COMPONENT_PHASE", "warm_all", len(jobs), flush=True)
        for job in jobs:
            for _ in range(args.warmup):
                body(job)
        tt.synchronize_device(mesh)
        cache_entries = mesh.num_program_cache_entries()
        addresses = [job["destination"].buffer_address() for job in jobs]
        for job in jobs:
            trace = tt.begin_trace_capture(mesh, cq_id=0)
            try:
                body(job)
            except BaseException:
                tt.end_trace_capture(mesh, trace, cq_id=0)
                tt.release_trace(mesh, trace)
                raise
            tt.end_trace_capture(mesh, trace, cq_id=0)
            traces.append(trace)
            job["trace"] = trace
        tt.synchronize_device(mesh)
        report["program_cache_entries_before_capture"] = cache_entries
        report["program_cache_entries_after_capture"] = mesh.num_program_cache_entries()
        assert mesh.num_program_cache_entries() == cache_entries, "Capture compiled an unwarmed program"
        report["trace_results"] = []
        print("REDUCTION_COMPONENT_PHASE", "replay", flush=True)
        for job in jobs:
            for _ in range(args.warmup):
                tt.execute_trace(mesh, job["trace"], cq_id=0, blocking=False)
            tt.synchronize_device(mesh)
            actual = read(job["destination"], job["n"])
            result = {
                "fixture": job["fixture"],
                "batch": job["n"],
                "strategy": job["strategy"],
                "trace_matches_eager": compare(job["expected"], actual),
            }
            assert result["trace_matches_eager"]["exact"], "Trace output differs from warmed eager output"
            if not args.tracked_correctness:
                seconds = []
                for _ in range(args.rounds):
                    tt.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        tt.execute_trace(mesh, job["trace"], cq_id=0, blocking=False)
                    tt.synchronize_device(mesh)
                    seconds.append((time.perf_counter() - start) / args.iterations)
                result["timing"] = {
                    "host_seconds_per_call_rounds": seconds,
                    "median_host_seconds_per_call": statistics.median(seconds),
                    "iterations_per_round": args.iterations,
                    "rounds": args.rounds,
                    "not_device_cycle_latency": True,
                }
            report["trace_results"].append(result)
        if not args.tracked_correctness:
            report["candidate_speedup_native_over_candidate"] = []
            for name, _ in fixtures:
                for n in (2, 32):
                    pair = {
                        row["strategy"]: row["timing"]["median_host_seconds_per_call"]
                        for row in report["trace_results"]
                        if row["fixture"] == name and row["batch"] == n
                    }
                    report["candidate_speedup_native_over_candidate"].append(
                        {"fixture": name, "batch": n, "speedup": pair["native"] / pair["candidate"]}
                    )
        assert addresses == [job["destination"].buffer_address() for job in jobs]
        assert mesh.num_program_cache_entries() == cache_entries
        report["immutable_inputs_and_persistent_destinations"] = True
        report["all_bodies_warmed_before_capture"] = True
        report["completed"] = True
    except BaseException:
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["candidate_sha256"] = hashlib.sha256(
            Path(__file__).with_name("probe_batch_reduction_counterfactual.py").read_bytes()
        ).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            for trace in traces:
                tt.release_trace(mesh, trace)
            tt.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
