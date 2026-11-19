# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warmed eager and traced complete corrected GEMM adapters; device opt-in."""
import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_fused_corrected_matmul_probe as fused
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source

FUSED_SHA = "845c959110a713f808d1475eb22ded5ffa4e6ddd8209f02170fb90ed02b353dc"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-device", action="store_true")
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--matrices", default="chunk0_gram,chunk0_transformed_v,chunk0_state_update")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--replays", type=int, default=100)
    p.add_argument("--eager-samples", type=int, default=7)
    p.add_argument("--max-cores", type=int, default=64)
    p.add_argument("--trace-region-size", type=int, default=64 * 1024 * 1024)
    p.add_argument("--order", choices=["baseline-first", "fused-first"], default="baseline-first")
    args = p.parse_args()
    assert not args.output.exists(), "Use fresh evidence path"
    assert args.warmup > 0 and args.samples > 0 and args.replays >= 100 and args.eager_samples > 0
    assert verify_source(fused.__file__, FUSED_SHA)
    checks = fused.source_checks()
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    result = dict(
        status="source_setup_pass",
        classification="Warmed complete GEMM adapter microbenchmark; not integrated decoder latency",
        script_sha256=fused.sha(__file__),
        helper_sha256=raw_sha256(fused.__file__),
        helper_historical_sha256=FUSED_SHA,
        checks=checks,
        device_executed=args.run_device,
        device_closed=False,
        settings=vars(args) | {"output": str(args.output)},
        timing_scope="Persistent identical DRAM FP32 inputs through complete baseline per-K slicing/WHERE/reshard/CorrectedFma and final DRAM output versus single generic operation to DRAM; setup/upload/readback excluded",
        timing=[],
        source_snapshots=[],
    )
    for path in (Path(__file__), Path(fused.__file__), fused.GRAPH_SOURCE, fused.ADAPTER_SOURCE):
        snapshot = Path(str(args.output) + "." + path.name + ".snapshot")
        assert not snapshot.exists()
        snapshot.write_bytes(path.read_bytes())
        result["source_snapshots"].append(dict(path=str(path), sha256=fused.sha(path), snapshot=str(snapshot)))

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    if not args.run_device:
        print("FUSED_MATMUL_TIMING_SOURCE_PASS", fused.sha(__file__))
        return
    for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_ALLOC_TRACKING"):
        assert os.environ.get(key) in (None, "", "0"), (key, os.environ.get(key))
    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_corrected_fma_matmul_probe import (
        CorrectedFma,
        CorrectedMatmul,
        compare,
        matrix_fixtures,
    )

    fixtures = matrix_fixtures(args.heads, set(args.matrices.split(",")), None)
    fixture_path = args.output.with_suffix(".fixtures.pt")
    torch.save(fixtures, fixture_path)
    result["fixtures"] = dict(path=str(fixture_path), sha256=fused.sha(fixture_path))
    save()
    mesh = None
    try:
        mesh = t.open_mesh_device(
            t.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768, trace_region_size=args.trace_region_size
        )
        baseline = CorrectedMatmul(CorrectedFma(t, mesh))
        candidate = fused.FusedCorrectedMatmul(t, mesh, args.max_cores)
        variants = {"baseline": baseline, "fused": candidate}
        order = ["baseline", "fused"] if args.order == "baseline-first" else ["fused", "baseline"]
        upload = lambda x: t.from_torch(
            x.contiguous(), device=mesh, dtype=t.float32, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
        )
        for fixture in fixtures:
            da, db = upload(fixture["a"]), upload(fixture["b"])
            with device_only():
                reference_tensor = baseline(da, db)
            t.synchronize_device(mesh)
            reference = t.to_torch(reference_tensor).clone()
            del reference_tensor
            for name in order:
                adapter = variants[name]
                for _ in range(args.warmup):
                    with device_only():
                        out = adapter(da, db)
                    t.synchronize_device(mesh)
                    del out
                eager = []
                for _ in range(args.eager_samples):
                    t.synchronize_device(mesh)
                    start = time.perf_counter()
                    with device_only():
                        out = adapter(da, db)
                    t.synchronize_device(mesh)
                    eager.append((time.perf_counter() - start) * 1000)
                    del out
                with device_only():
                    out = adapter(da, db)
                eager_check = compare(reference, t.to_torch(out))
                assert eager_check["fp32_bits_different"] == 0, eager_check
                del out
                trace_id = None
                capture_open = False
                output = None
                try:
                    trace_id = t.begin_trace_capture(mesh, cq_id=0)
                    capture_open = True
                    with device_only():
                        output = adapter(da, db)
                    t.end_trace_capture(mesh, trace_id, cq_id=0)
                    capture_open = False
                    trace_allocation_tracker.acknowledge_corruptible(output)
                    for _ in range(args.warmup):
                        t.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                    first = t.to_torch(output).clone()
                    first_check = compare(reference, first)
                    assert first_check["fp32_bits_different"] == 0, first_check
                    samples = []
                    for _ in range(args.samples):
                        t.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            t.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                        t.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays)
                    final = t.to_torch(output).clone()
                    last_check = compare(reference, final)
                    determinism = compare(first, final)
                    assert last_check["fp32_bits_different"] == determinism["fp32_bits_different"] == 0
                    row = dict(
                        matrix=fixture["name"],
                        variant=name,
                        a_shape=list(fixture["a"].shape),
                        b_shape=list(fixture["b"].shape),
                        warmed_eager_median_ms=statistics.median(eager),
                        warmed_eager_samples_ms=eager,
                        traced_median_ms=statistics.median(samples),
                        traced_samples_ms=samples,
                        replays_per_sample=args.replays,
                        timed_replays=args.replays * args.samples,
                        first_vs_baseline=first_check,
                        last_vs_baseline=last_check,
                        first_vs_last=determinism,
                        eager_vs_baseline=eager_check,
                        output_memory=str(output.memory_config()),
                    )
                    result["timing"].append(row)
                    save()
                    print(json.dumps(row), flush=True)
                finally:
                    if trace_id is not None:
                        try:
                            if capture_open:
                                t.end_trace_capture(mesh, trace_id, cq_id=0)
                            t.synchronize_device(mesh)
                        finally:
                            t.release_trace(mesh, trace_id)
                    del output
            del da, db
        pairs = {}
        for row in result["timing"]:
            pairs.setdefault(row["matrix"], {})[row["variant"]] = row
        result["comparisons"] = [
            dict(
                matrix=matrix,
                eager_speedup=rows["baseline"]["warmed_eager_median_ms"] / rows["fused"]["warmed_eager_median_ms"],
                traced_speedup=rows["baseline"]["traced_median_ms"] / rows["fused"]["traced_median_ms"],
            )
            for matrix, rows in pairs.items()
        ]
        result["status"] = "completed_microbenchmark"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


if __name__ == "__main__":
    main()
