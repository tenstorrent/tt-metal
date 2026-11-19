# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Matched transpose-matmul timings with reversed capture and measurement order."""

import argparse
import json
import os
import statistics
import time
import traceback
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_corrected_matmul_transpose_probe as component

COMPONENT_SHA = "e48c3ba24c24be0099c7b45eb2fc0ffeb98a0c90d4181b015e110c8236b9e767"


def device(args, fixtures, base_scope, candidate_scope, result, save):
    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    for var in (
        "TT_VISIBLE_DEVICES",
        "TT_METAL_WATCHER",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_TRACE_ALLOC_TRACKING",
        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
    ):
        assert var not in os.environ, var
    mesh = None
    try:
        mesh = t.open_mesh_device(
            t.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=67108864, l1_small_size=32768
        )
        base = base_scope["FusedCorrectedMatmul"](t, mesh)
        candidate = candidate_scope["TransposeCorrectedMatmul"](t, mesh)
        for fixture in fixtures:
            upload = lambda value: t.from_torch(
                value, device=mesh, dtype=t.float32, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
            )
            a, b = upload(fixture["a"]), upload(fixture["b"])
            ta, tb = fixture["transpose_a"], fixture["transpose_b"]

            def call(label):
                with device_only():
                    if label == "candidate":
                        return candidate(a, b, transpose_a=ta, transpose_b=tb)
                    return base(t.transpose(a, -1, -2) if ta else a, t.transpose(b, -1, -2) if tb else b)

            expected = t.to_torch(call("baseline"))
            for capture_order in (("candidate", "baseline"), ("baseline", "candidate")):
                row = dict(
                    name=fixture["name"],
                    capture_order=capture_order,
                    input_addresses=[a.buffer_address(), b.buffer_address()],
                    shapes=[list(a.shape), list(b.shape)],
                    transpose_flags=[ta, tb],
                    eager_samples_ms={"baseline": [], "candidate": []},
                    trace_samples_ms={"baseline": [], "candidate": []},
                    measurement_order=[],
                    trace_outputs={},
                    exact_checks=[],
                )
                result["cases"].append(row)
                # Compile/warm both complete adapters before either measurement; alternate order.
                for i in range(3):
                    for label in capture_order if i % 2 == 0 else capture_order[::-1]:
                        out = call(label)
                        assert component.bitdiff(expected, t.to_torch(out))["exact"]
                for i in range(args.samples):
                    order = capture_order if i % 2 == 0 else capture_order[::-1]
                    for label in order:
                        t.synchronize_device(mesh)
                        start = time.perf_counter()
                        out = call(label)
                        t.synchronize_device(mesh)
                        row["eager_samples_ms"][label].append((time.perf_counter() - start) * 1000)
                        assert component.bitdiff(expected, t.to_torch(out))["exact"]
                traces, outputs = {}, {}
                try:
                    for label in capture_order:
                        tid = t.begin_trace_capture(mesh, cq_id=0)
                        outputs[label] = call(label)
                        t.end_trace_capture(mesh, tid, cq_id=0)
                        traces[label] = tid
                        row["trace_outputs"][label] = outputs[label].buffer_address()
                    # Both trace allocations remain live during all paired samples.
                    for label in capture_order:
                        t.execute_trace(mesh, traces[label], cq_id=0, blocking=True)
                        assert component.bitdiff(expected, t.to_torch(outputs[label]))["exact"]
                    for i in range(args.samples):
                        order = capture_order if i % 2 == 0 else capture_order[::-1]
                        row["measurement_order"].append(order)
                        for label in order:
                            t.synchronize_device(mesh)
                            start = time.perf_counter()
                            for _ in range(args.replays):
                                t.execute_trace(mesh, traces[label], cq_id=0, blocking=False)
                            t.synchronize_device(mesh)
                            row["trace_samples_ms"][label].append((time.perf_counter() - start) * 1000 / args.replays)
                            check = component.bitdiff(expected, t.to_torch(outputs[label]))
                            row["exact_checks"].append(dict(sample=i, adapter=label, **check))
                            assert check["exact"]
                        save()
                    row["traced_median_ms"] = {k: statistics.median(v) for k, v in row["trace_samples_ms"].items()}
                    row["eager_median_ms"] = {k: statistics.median(v) for k, v in row["eager_samples_ms"].items()}
                    row["paired_trace_candidate_minus_baseline_ms"] = [
                        c - b for b, c in zip(row["trace_samples_ms"]["baseline"], row["trace_samples_ms"]["candidate"])
                    ]
                    row["plan"] = candidate.last_plan
                    row["replays_per_adapter"] = args.samples * args.replays
                finally:
                    for tid in traces.values():
                        t.release_trace(mesh, tid)
                outputs.clear()
                save()
            t.deallocate(a)
            t.deallocate(b)
        result["status"] = "matched_timing_exact"
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-device", action="store_true")
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--replays", type=int, default=20)
    args = p.parse_args()
    assert not args.output.exists() and args.samples >= 2 and args.samples % 2 == 0 and args.replays > 0
    assert component.sha(component.__file__) == COMPONENT_SHA
    gate = component.E / "v6_corrected_matmul_transpose_device.json"
    proof = json.loads(gate.read_text())
    assert proof["status"] == "pass" and proof["device_closed"] and len(proof["device_cases"]) == 22
    assert proof["script_sha256"] == COMPONENT_SHA and all(c["output"]["exact"] for c in proof["device_cases"])
    base, candidate, source_proof, _, _ = component.source_bundle()
    fixtures = [c for c in component.fixtures() if c["timing"]]
    # Reverse the original site order as well: anomaly first, Gram last.
    fixtures.reverse()
    result = dict(
        status="source_setup_pass",
        source_sha256=component.sha(__file__),
        component_sha256=COMPONENT_SHA,
        component_gate=dict(path=str(gate), sha256=component.sha(gate)),
        source_proof=source_proof,
        fixture_order=[c["name"] for c in fixtures],
        samples=args.samples,
        replays=args.replays,
        cases=[],
        device_closed=False,
        classification="Matched complete-adapter timing, no whole-decoder claim",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    args.output.with_suffix(".source.py.snapshot").write_bytes(Path(__file__).read_bytes())
    save()
    if args.run_device:
        try:
            device(args, fixtures, base, candidate, result, save)
        except Exception:
            result.update(status="failed", error=traceback.format_exc())
            save()
            raise
    assert component.sha(__file__) == result["source_sha256"]
    save()
    print(result["status"], result["source_sha256"], flush=True)


if __name__ == "__main__":
    main()
