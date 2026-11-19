# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: complete pair-gather -> tiled projection coefficient adapters.

Each adapter starts at BF16 TILE [1,1,2048] in DRAM and returns BF16 TILE
[2048/K,K,1,1] in DRAM. All runtime layout conversions and reshapes are timed.
This microprobe cannot satisfy decoder accuracy or performance acceptance.
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

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder"
VARIANTS = ("original_tile", "compact_tile", "original_rm", "compact_rm")


def inputs(seed, random_cases):
    generator = torch.Generator().manual_seed(seed)
    cases = []
    for _ in range(random_cases):
        x = torch.randn((1, 1, 2048), generator=generator, dtype=torch.bfloat16)
        # Representative normalized input; no checkpoint or HF runtime is needed.
        x = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
        cases.append(x)
    # Unique finite BF16 codes make every mistaken permutation observable.
    cases.append(
        (torch.arange(2048, dtype=torch.int32) + 13000).to(torch.uint16).view(torch.bfloat16).reshape(1, 1, 2048)
    )
    return cases


def compare(expected, actual):
    a = expected.contiguous().view(torch.int16).reshape(-1)
    b = actual.bfloat16().contiguous().view(torch.int16).reshape(-1)
    assert a.shape == b.shape
    bad = torch.nonzero(a != b).flatten()
    return dict(
        elements=a.numel(),
        code_differences=bad.numel(),
        exact=bad.numel() == 0,
        max_abs=float((expected.float().reshape(-1) - actual.float().reshape(-1)).abs().max()),
        first_differences=[
            dict(index=int(i), expected_code=int(a[i]) & 65535, actual_code=int(b[i]) & 65535) for i in bad[:8]
        ],
    )


def run(args):
    torch.set_num_threads(4)
    path = Path(args.output)
    assert args.overwrite or not path.exists(), f"Preserve existing evidence: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    values = inputs(args.seed, args.random_cases)
    order = torch.arange(2048).reshape(1024, 2).flip(1).reshape(1, 1, 2048)
    pair32 = torch.arange(32).reshape(16, 2).flip(1).reshape(1, 1, 32).expand(1, 64, 32).contiguous()
    cpu_checks = []
    for k in (512, 1024):
        for x in values:
            expected = torch.gather(x, -1, order).reshape(2048 // k, k, 1, 1)
            compact = torch.gather(x.reshape(1, 64, 32), -1, pair32).reshape(expected.shape)
            cpu_checks.append(compare(expected, compact))
    assert all(c["exact"] for c in cpu_checks)
    snapshot = DOC / "endurance/compact_pair_gather.py.snapshot"
    result = dict(
        classification="diagnostic_only_not_acceptance",
        status="cpu_pass",
        seed=args.seed,
        random_cases=args.random_cases,
        unique_code_case=True,
        input_contract="BF16 TILE DRAM [1,1,2048]",
        output_contract="BF16 TILE DRAM [2048/K,K,1,1]",
        timing_scope="Complete gather adapter including DRAM placement, reshapes, untilize/tilize; no projection multiply/reduction",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        candidate_snapshot=str(snapshot),
        candidate_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        input_sha256=[hashlib.sha256(x.view(torch.uint16).numpy().tobytes()).hexdigest() for x in values],
        cpu_checks=cpu_checks,
        variants=[],
    )

    def save():
        path.write_text(json.dumps(result, indent=2) + "\n")

    if args.oracle_only:
        save()
        print("CPU_PAIR_GATHER_PROOF_PASS", len(cpu_checks))
        return

    for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_ALLOC_TRACKING"):
        assert os.environ.get(key) in (None, "", "0"), f"Clean adapter timing requires {key} disabled"
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    record(path)
    parameters = dict(physical_device_ids=[0], trace_region_size=16 * 1024 * 1024)
    result.update(status="running", device_open_parameters=parameters)
    save()
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **parameters)

        def upload(x, dtype, layout):
            return ttnn.from_torch(
                x.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        device_inputs = [upload(x, ttnn.bfloat16, ttnn.TILE_LAYOUT) for x in values]
        indices = {
            (compact, rm): upload(
                pair32 if compact else order, ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT
            )
            for compact in (False, True)
            for rm in (False, True)
        }

        def adapter(x, variant, k):
            compact, rm = variant.startswith("compact"), variant.endswith("rm")
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            if rm:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            if compact:
                x = ttnn.reshape(x, (1, 64, 32))
            ordered = ttnn.gather(x, -1, indices[compact, rm])
            coefficients = ttnn.reshape(ordered, (2048 // k, k, 1, 1))
            if rm:
                coefficients = ttnn.to_layout(coefficients, ttnn.TILE_LAYOUT)
            return coefficients

        def validate_output(value, expected):
            assert tuple(value.shape) == tuple(expected.shape)
            assert value.dtype == ttnn.bfloat16 and value.layout == ttnn.TILE_LAYOUT
            assert value.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            return compare(expected, ttnn.to_torch(value))

        for k in args.projection_k:
            reference = [torch.gather(x, -1, order).reshape(2048 // k, k, 1, 1) for x in values]
            for variant in args.variants:
                row = dict(name=variant, projection_k=k, status="running", eager=[])
                result["variants"].append(row)
                trace_id, capture_open = None, False
                out = None
                stage = "eager"
                try:
                    for x, expected in zip(device_inputs, reference):
                        with device_only():
                            out = adapter(x, variant, k)
                        row["eager"].append(validate_output(out, expected))
                        del out
                        out = None
                    assert all(c["exact"] for c in row["eager"]), "Coefficient code mismatch; timing skipped"
                    stage = "warmup"
                    for _ in range(args.warmup):
                        with device_only():
                            out = adapter(device_inputs[0], variant, k)
                        ttnn.synchronize_device(mesh)
                        del out
                        out = None
                    stage = "capture"
                    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                    capture_open = True
                    with device_only():
                        out = adapter(device_inputs[0], variant, k)
                    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                    capture_open = False
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                    row["first_replay"] = validate_output(out, reference[0])
                    stage = "timing"
                    samples = []
                    for _ in range(args.samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays_per_sample):
                            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays_per_sample)
                    row["last_replay"] = validate_output(out, reference[0])
                    row.update(
                        samples_ms=samples,
                        median_ms=statistics.median(samples),
                        min_ms=min(samples),
                        replays_per_sample=args.replays_per_sample,
                        synchronization="sample boundaries; async queued trace replays",
                        warmup=args.warmup,
                    )
                    assert row["first_replay"]["exact"] and row["last_replay"]["exact"]
                    row["status"] = "exact"
                except Exception:
                    row.update(status="failed", failure_stage=stage, error=traceback.format_exc())
                finally:
                    if trace_id is not None:
                        if capture_open:
                            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                        ttnn.synchronize_device(mesh)
                        ttnn.release_trace(mesh, trace_id)
                    del out
                    out = None
                    save()
        for row in result["variants"]:
            baseline = next(
                (
                    r
                    for r in result["variants"]
                    if r["name"] == "original_tile"
                    and r["projection_k"] == row["projection_k"]
                    and r["status"] == "exact"
                ),
                None,
            )
            if baseline is not None and row["status"] == "exact":
                row["speedup_vs_original_complete_adapter"] = baseline["median_ms"] / row["median_ms"]
        result["status"] = (
            "complete" if all(r["status"] == "exact" for r in result["variants"]) else "completed_with_failures"
        )
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        save()
    print("PAIR_GATHER_ADAPTER_DIAGNOSTIC", result["status"], path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DOC / "endurance/compact_pair_gather_probe.json"))
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--random-cases", type=int, default=3)
    parser.add_argument("--projection-k", type=int, nargs="+", choices=(512, 1024), default=[512, 1024])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--replays-per-sample", type=int, default=20)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    assert args.random_cases >= 1 and args.samples > 0 and args.warmup > 0 and args.replays_per_sample > 0
    run(args)
