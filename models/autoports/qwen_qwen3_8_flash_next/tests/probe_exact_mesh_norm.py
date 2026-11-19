# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optional local2560 merged-RHS norm control; no production dispatch changes."""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch
from ttnn.tools import trace_allocation_tracker as tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_exact_residual_write import fixtures, read_ranks
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import rms
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import merged_rms
from models.autoports.qwen_qwen3_8_flash_next.tt.model import Checkpoint
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshLoader


def bits_equal(actual, expected):
    assert len(actual) == len(expected) == 4
    rows = []
    for rank, (a, b) in enumerate(zip(actual, expected, strict=True)):
        assert a.dtype == b.dtype and a.shape == b.shape
        integer = torch.int32 if a.dtype == torch.float32 else torch.int16
        rows.append({"rank": rank, "exact": torch.equal(a.view(integer), b.view(integer))})
    assert all(r["exact"] for r in rows), rows
    return rows


def scales(value, ones, eps):
    """Separate diagnostic outputs; excluded from timed norm captures."""
    widened = ttnn.typecast(value, ttnn.float32)
    variance = ttnn.mean(ttnn.multiply(widened, widened), dim=-1, keepdim=True)
    baseline = ttnn.rsqrt(ttnn.add(variance, eps))
    add_rsqrt = ttnn.add(variance, eps, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT, 0.0)])
    rhs = ttnn.multiply(
        ones,
        variance,
        input_tensor_b_activations=[
            ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, eps),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT, 0.0),
        ],
    )
    return baseline, add_rsqrt, rhs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=os.environ.get("QWEN_CHECKPOINT"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-timing", action="store_true")
    parser.add_argument(
        "--include-add-rsqrt", action="store_true", help="Include previously slower exact single-chip control"
    )
    parser.add_argument("--repeats", type=int, default=64)
    args = parser.parse_args()
    assert args.snapshot is not None and args.snapshot.is_dir()
    assert args.repeats > 0
    assert bool(tracker.TRACE_ALLOC_TRACKING) != args.clean_timing
    assert not args.clean_timing or not os.environ.get("TT_METAL_WATCHER")
    torch.set_num_threads(8)
    checkpoint = Checkpoint(args.snapshot)
    eps = 1e-6
    keys = (
        "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight",
        "model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
    )
    weights = [(key, checkpoint.tensor(key).float().reshape(4, 2560) + 1) for key in keys]
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        root / "tt/fused_decoder.py",
        root / "tt/decoder_components.py",
        root / "tests/probe_exact_residual_write.py",
    ]
    report = {
        "passed": False,
        "rows": [],
        "mesh": [1, 4],
        "snapshot": str(args.snapshot),
        "eps": eps,
        "scope": "Real FP32 norm weights, synthetic actual-shape residuals; not full-model acceptance",
        "tracking": bool(tracker.TRACE_ALLOC_TRACKING),
        "clean_timing": args.clean_timing,
        "watcher": os.environ.get("TT_METAL_WATCHER"),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "boundary": "Complete norm trace only; no scale diagnostics or readbacks in timed nonblocking replay windows",
        "prior_disposition": "norm_rhs individually exact; prior aggregate JSON fails unrelated matmul_activation. add_rsqrt exact but tracked single-chip decode slower.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh, trace = None, None

    def phase(label):
        report["last_phase"] = label
        print(f"EXACT_NORM_PHASE {label}", flush=True)

    try:
        phase("open_mesh")
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        loader = MeshLoader({}, mesh)
        weight_tensors = [
            (key, loader.shards([v.reshape(1, 1, 1, 2560) for v in weight], dtype=ttnn.float32))
            for key, weight in weights
        ]
        for count in (1, 3, 17, 32):
            phase(f"q{count}:setup_and_scale_checks")
            host_families = [[rank[0] for rank in family] for family in fixtures(count)]
            host_families.append([values * (rank + 1) * 0.0009765625 for rank, values in enumerate(host_families[0])])
            inputs = [loader.shards(family) for family in host_families]
            ones = loader.shards([torch.ones(1, 1, count, 1) for _ in range(4)], dtype=ttnn.float32)
            scale_checks = []
            for value in inputs:
                with runtime_audit():
                    baseline_scale, add_scale, rhs_scale = scales(value, ones, eps)
                expected_scale = read_ranks(baseline_scale)
                checks = {"rhs": bits_equal(read_ranks(rhs_scale), expected_scale)}
                if args.include_add_rsqrt:
                    checks["add_rsqrt"] = bits_equal(read_ranks(add_scale), expected_scale)
                scale_checks.append(checks)
                del baseline_scale, add_scale, rhs_scale
            live = ttnn.empty_like(inputs[0])
            for key, weight in weight_tensors:
                phase(f"q{count}:{key}:baseline_eager")
                expected = []
                for value in inputs:
                    with runtime_audit():
                        output = rms(value, weight, eps)
                    expected.append(read_ranks(output))
                    del output
                variants = {
                    "baseline": lambda value: rms(value, weight, eps),
                    "rhs": lambda value: merged_rms(value, weight, eps, rhs=True),
                }
                if args.include_add_rsqrt:
                    variants["add_rsqrt"] = lambda value: merged_rms(value, weight, eps, rhs=False)
                row = {"count": count, "weight": key, "scale_checks": scale_checks, "variants": {}}
                report["rows"].append(row)
                for name, function in variants.items():
                    phase(f"q{count}:{key}:{name}:eager")
                    item = {"eager": [], "changed_replays": [], "windows_ms_per_replay": []}
                    row["variants"][name] = item
                    for family, value in enumerate(inputs):
                        with runtime_audit():
                            output = function(value)
                        item["eager"].append(bits_equal(read_ranks(output), expected[family]))
                        del output
                    ttnn.copy(inputs[0], live)
                    with runtime_audit():
                        warm = function(live)
                    del warm
                    gc.collect()
                    phase(f"q{count}:{key}:{name}:capture")
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    try:
                        with runtime_audit():
                            output = function(live)
                    finally:
                        ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    if tracker.TRACE_ALLOC_TRACKING:
                        tracker.acknowledge_corruptible(output)
                    phase(f"q{count}:{key}:{name}:changed_input_replay")
                    for repeat in range(12):
                        family = repeat % len(inputs)
                        ttnn.copy(inputs[family], live)
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        item["changed_replays"].append(bits_equal(read_ranks(output), expected[family]))
                    phase(f"q{count}:{key}:{name}:timing")
                    for _ in range(3):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.repeats):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        item["windows_ms_per_replay"].append((time.perf_counter() - start) * 1000 / args.repeats)
                        bits_equal(read_ranks(output), expected[2])
                    item["median_ms"] = statistics.median(item["windows_ms_per_replay"])
                    item["repeats_per_window"] = args.repeats
                    ttnn.release_trace(mesh, trace)
                    trace = None
                    del output
                print(
                    json.dumps(
                        {"count": count, "key": key, "ms": {k: v["median_ms"] for k, v in row["variants"].items()}}
                    ),
                    flush=True,
                )
            del live, inputs, ones
        report["passed"] = True
        phase("passed")
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
