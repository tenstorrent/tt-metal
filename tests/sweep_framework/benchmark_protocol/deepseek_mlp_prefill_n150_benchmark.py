#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Micro-benchmark for DeepSeek MLP prefill program config generation on N150.

This benchmark intentionally exercises MLP._get_prefill_pc() and runs the
resulting program_config via ttnn.matmul on N150-safe DeepSeek-oriented shapes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import ttnn

from models.demos.deepseek_v3.tt.mlp.mlp import MLP


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    w = idx - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def _load_cases(cases_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No cases found in {cases_path}")
    return cases


def _make_tensors_for_case(case: dict[str, Any], device: ttnn.Device) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    seq_len = int(case["seq_len"])
    dim = int(case["dim"])
    hidden_dim = int(case["hidden_dim"])
    is_w2 = bool(case["is_w2"])

    if is_w2:
        in_features = hidden_dim
        out_features = dim
    else:
        in_features = dim
        out_features = hidden_dim

    a = torch.randn((1, 1, seq_len, in_features), dtype=torch.bfloat16)
    b = torch.randn((1, 1, in_features, out_features), dtype=torch.bfloat16)

    tt_a = ttnn.from_torch(
        a,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_b = ttnn.from_torch(
        b,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tt_a, tt_b


def _run_case(
    case: dict[str, Any],
    device: ttnn.Device,
    warmup_iters: int,
    measure_iters: int,
) -> dict[str, Any]:
    seq_len = int(case["seq_len"])
    dim = int(case["dim"])
    hidden_dim = int(case["hidden_dim"])
    is_w2 = bool(case["is_w2"])

    core_grid_coord = ttnn.CoreCoord(device.core_grid.x, device.core_grid.y)
    program_config = MLP._get_prefill_pc(
        seq_len=seq_len,
        dim=dim,
        hidden_dim=hidden_dim,
        num_devices=1,
        core_grid_size=core_grid_coord,
        is_w2=is_w2,
    )

    tt_a, tt_b = _make_tensors_for_case(case, device)

    for _ in range(warmup_iters):
        out = ttnn.matmul(tt_a, tt_b, program_config=program_config)
        _ = ttnn.to_torch(out)

    samples_ms: list[float] = []
    for _ in range(measure_iters):
        start_ns = time.time_ns()
        out = ttnn.matmul(tt_a, tt_b, program_config=program_config)
        _ = ttnn.to_torch(out)
        elapsed_ms = (time.time_ns() - start_ns) / 1_000_000.0
        samples_ms.append(elapsed_ms)

    samples_sorted = sorted(samples_ms)
    p50 = _percentile(samples_sorted, 0.50)
    p95 = _percentile(samples_sorted, 0.95)

    return {
        "name": case["name"],
        "seq_len": seq_len,
        "dim": dim,
        "hidden_dim": hidden_dim,
        "is_w2": is_w2,
        "program_config": {
            "compute_with_storage_grid_size": [
                int(program_config.compute_with_storage_grid_size.x),
                int(program_config.compute_with_storage_grid_size.y),
            ],
            "per_core_M": int(program_config.per_core_M),
            "per_core_N": int(program_config.per_core_N),
            "in0_block_w": int(program_config.in0_block_w),
            "out_subblock_h": int(program_config.out_subblock_h),
            "out_subblock_w": int(program_config.out_subblock_w),
        },
        "perf_ms": {
            "mean": statistics.fmean(samples_ms),
            "p50": p50,
            "p95": p95,
            "samples": samples_ms,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek MLP prefill PC benchmark on N150-safe cases")
    parser.add_argument(
        "--cases-json",
        type=Path,
        default=Path(
            "/home/ubuntu/tt-metal/tests/sweep_framework/benchmark_protocol/manifests/deepseek_mlp_prefill_n150_cases.json"
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "/home/ubuntu/tt-metal/tests/sweep_framework/benchmark_protocol/generated/deepseek_oriented/deepseek_mlp_prefill_n150_benchmark_last.json"
        ),
    )
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=10)
    args = parser.parse_args()

    cases = _load_cases(args.cases_json)

    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
    try:
        results = [
            _run_case(
                case=case,
                device=device,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
            )
            for case in cases
        ]
    finally:
        ttnn.close_device(device)

    summary_p50s = sorted(r["perf_ms"]["p50"] for r in results if r["perf_ms"]["p50"] is not None)
    summary_p95s = sorted(r["perf_ms"]["p95"] for r in results if r["perf_ms"]["p95"] is not None)

    payload = {
        "cases_json": str(args.cases_json),
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "results": results,
        "summary": {
            "case_count": len(results),
            "overall_p50_ms": _percentile(summary_p50s, 0.50) if summary_p50s else None,
            "overall_p95_ms": _percentile(summary_p95s, 0.95) if summary_p95s else None,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote benchmark JSON -> {args.out_json}")
    print(f"Cases: {payload['summary']['case_count']}")
    print(f"overall_p50_ms: {payload['summary']['overall_p50_ms']}")
    print(f"overall_p95_ms: {payload['summary']['overall_p95_ms']}")
    for row in results:
        grid = row["program_config"]["compute_with_storage_grid_size"]
        print(
            f"- {row['name']}: grid={grid}, per_core_M={row['program_config']['per_core_M']}, "
            f"per_core_N={row['program_config']['per_core_N']}, p50={row['perf_ms']['p50']:.3f}ms, "
            f"p95={row['perf_ms']['p95']:.3f}ms"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
