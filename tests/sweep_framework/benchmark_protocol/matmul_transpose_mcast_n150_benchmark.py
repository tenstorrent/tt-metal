#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import ttnn


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


def _run_case(device: ttnn.Device, case: dict, warmup_iters: int, measure_iters: int) -> dict:
    m = int(case["m"])
    k = int(case["k"])
    n = int(case["n"])
    grid_x = int(case["grid_x"])
    grid_y = int(case["grid_y"])

    core_grid = ttnn.CoreGrid(x=grid_x, y=grid_y)
    in0_memcfg = ttnn.create_sharded_memory_config(
        (1, 1, m, k),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
    )

    torch_a = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((1, 1, k, n), dtype=torch.bfloat16)
    golden = torch.matmul(torch_a, torch_b)

    a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memcfg,
    )
    b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for _ in range(warmup_iters):
        y = ttnn.matmul(
            a,
            b,
            core_grid=core_grid,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _ = ttnn.to_torch(y)

    samples_ms: list[float] = []
    last_out = None
    for _ in range(measure_iters):
        start_ns = time.time_ns()
        y = ttnn.matmul(
            a,
            b,
            core_grid=core_grid,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        last_out = ttnn.to_torch(y)
        samples_ms.append((time.time_ns() - start_ns) / 1_000_000.0)

    pcc_ok = bool(torch.allclose(last_out, golden, rtol=1e-1, atol=1e-1))
    sorted_samples = sorted(samples_ms)
    return {
        "name": case["name"],
        "shape": [m, k, n],
        "core_grid": [grid_x, grid_y],
        "pcc_ok": pcc_ok,
        "perf_ms": {
            "mean": statistics.fmean(samples_ms),
            "p50": _percentile(sorted_samples, 0.50),
            "p95": _percentile(sorted_samples, 0.95),
            "samples": samples_ms,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Transpose-mcast auto matmul benchmark on N150")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=10)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "/home/ubuntu/tt-metal/tests/sweep_framework/benchmark_protocol/generated/deepseek_oriented/matmul_transpose_mcast_n150_benchmark_last.json"
        ),
    )
    args = parser.parse_args()

    cases = [
        {"name": "txm_non_square_grid_8x4_m256_k1024_n4096", "m": 256, "k": 1024, "n": 4096, "grid_x": 8, "grid_y": 4},
        {"name": "txm_non_square_grid_8x2_m256_k1024_n4096", "m": 256, "k": 1024, "n": 4096, "grid_x": 8, "grid_y": 2},
    ]

    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
    try:
        rows = [_run_case(device, c, args.warmup_iters, args.measure_iters) for c in cases]
    finally:
        ttnn.close_device(device)

    p50s = sorted(r["perf_ms"]["p50"] for r in rows if r["perf_ms"]["p50"] is not None)
    p95s = sorted(r["perf_ms"]["p95"] for r in rows if r["perf_ms"]["p95"] is not None)
    payload = {
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "results": rows,
        "summary": {
            "case_count": len(rows),
            "all_pcc_ok": all(r["pcc_ok"] for r in rows),
            "overall_p50_ms": _percentile(p50s, 0.50),
            "overall_p95_ms": _percentile(p95s, 0.95),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote benchmark JSON -> {args.out_json}")
    print(f"overall_p50_ms: {payload['summary']['overall_p50_ms']}")
    print(f"overall_p95_ms: {payload['summary']['overall_p95_ms']}")
    print(f"all_pcc_ok: {payload['summary']['all_pcc_ok']}")
    for r in rows:
        print(
            f"- {r['name']}: grid={r['core_grid']} p50={r['perf_ms']['p50']:.3f}ms "
            f"p95={r['perf_ms']['p95']:.3f}ms pcc_ok={r['pcc_ok']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
