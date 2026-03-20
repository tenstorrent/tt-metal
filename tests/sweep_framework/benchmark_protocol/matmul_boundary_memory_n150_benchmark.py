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


def _choose_intermediate_memcfg(mode: str, seq_len: int, dim: int, hidden_dim: int) -> ttnn.MemoryConfig:
    if mode == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    if mode == "l1":
        return ttnn.L1_MEMORY_CONFIG
    # Conservative adaptive rule:
    # keep DRAM for larger boundaries and use L1 only for moderate hidden sizes at short sequence.
    if seq_len <= 64 and dim <= 1536 and hidden_dim <= 3072:
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def _run_case(case: dict, device: ttnn.Device, inter_mem_mode: str, warmup_iters: int, measure_iters: int) -> dict:
    seq_len = int(case["seq_len"])
    dim = int(case["dim"])
    hidden_dim = int(case["hidden_dim"])
    inter_memcfg = _choose_intermediate_memcfg(inter_mem_mode, seq_len, dim, hidden_dim)

    a = ttnn.from_torch(
        torch.randn((1, 1, seq_len, dim), dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w1 = ttnn.from_torch(
        torch.randn((1, 1, dim, hidden_dim), dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w2 = ttnn.from_torch(
        torch.randn((1, 1, hidden_dim, dim), dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for _ in range(warmup_iters):
        y = ttnn.matmul(a, w1, memory_config=inter_memcfg)
        z = ttnn.matmul(y, w2, memory_config=inter_memcfg)
        _ = ttnn.to_torch(z)

    samples_ms: list[float] = []
    for _ in range(measure_iters):
        start_ns = time.time_ns()
        y = ttnn.matmul(a, w1, memory_config=inter_memcfg)
        z = ttnn.matmul(y, w2, memory_config=inter_memcfg)
        _ = ttnn.to_torch(z)
        samples_ms.append((time.time_ns() - start_ns) / 1_000_000.0)

    sorted_samples = sorted(samples_ms)
    return {
        "name": case["name"],
        "seq_len": seq_len,
        "dim": dim,
        "hidden_dim": hidden_dim,
        "intermediate_memory": "l1" if inter_memcfg == ttnn.L1_MEMORY_CONFIG else "dram",
        "perf_ms": {
            "mean": statistics.fmean(samples_ms),
            "p50": _percentile(sorted_samples, 0.50),
            "p95": _percentile(sorted_samples, 0.95),
            "samples": samples_ms,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Matmul boundary memory benchmark on N150")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=10)
    parser.add_argument("--intermediate-memory", choices=["dram", "l1", "adaptive"], default="dram")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "/home/ubuntu/tt-metal/tests/sweep_framework/benchmark_protocol/generated/deepseek_oriented/matmul_boundary_memory_n150_last.json"
        ),
    )
    args = parser.parse_args()

    cases = [
        {"name": "mid_2048_4096", "seq_len": 32, "dim": 2048, "hidden_dim": 4096},
        {"name": "mid_1536_3072", "seq_len": 32, "dim": 1536, "hidden_dim": 3072},
        {"name": "small_1024_2048", "seq_len": 32, "dim": 1024, "hidden_dim": 2048},
    ]

    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
    try:
        rows = [
            _run_case(case, device, args.intermediate_memory, args.warmup_iters, args.measure_iters) for case in cases
        ]
    finally:
        ttnn.close_device(device)

    p50s = sorted(r["perf_ms"]["p50"] for r in rows if r["perf_ms"]["p50"] is not None)
    p95s = sorted(r["perf_ms"]["p95"] for r in rows if r["perf_ms"]["p95"] is not None)
    payload = {
        "intermediate_memory": args.intermediate_memory,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "results": rows,
        "summary": {
            "case_count": len(rows),
            "overall_p50_ms": _percentile(p50s, 0.50),
            "overall_p95_ms": _percentile(p95s, 0.95),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote benchmark JSON -> {args.out_json}")
    print(f"intermediate_memory: {args.intermediate_memory}")
    print(f"overall_p50_ms: {payload['summary']['overall_p50_ms']}")
    print(f"overall_p95_ms: {payload['summary']['overall_p95_ms']}")
    for row in rows:
        print(f"- {row['name']}: p50={row['perf_ms']['p50']:.3f}ms " f"p95={row['perf_ms']['p95']:.3f}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
