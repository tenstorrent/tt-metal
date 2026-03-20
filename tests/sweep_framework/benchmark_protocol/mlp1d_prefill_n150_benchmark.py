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

from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.mlp.mlp_1d import MLP1DConfig, _resolve_mlp1d_config


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


def _build_resolved_cfg(device: ttnn.Device, dim: int, hidden_dim: int) -> MLP1DConfig:
    w1 = LazyWeight(source=torch.randn((dim, hidden_dim), dtype=torch.bfloat16))
    w2 = LazyWeight(source=torch.randn((hidden_dim, dim), dtype=torch.bfloat16))
    w3 = LazyWeight(source=torch.randn((dim, hidden_dim), dtype=torch.bfloat16))
    cfg = MLP1DConfig(w1=w1, w2=w2, w3=w3, mesh_device=device, dim=dim, hidden_dim=hidden_dim)
    return _resolve_mlp1d_config(cfg)


def _run_single(case: dict, device: ttnn.Device, warmup_iters: int, measure_iters: int) -> dict:
    seq_len = int(case["seq_len"])
    dim = int(case["dim"])
    hidden_dim = int(case["hidden_dim"])

    cfg = _build_resolved_cfg(device, dim=dim, hidden_dim=hidden_dim)
    pc_w1 = cfg.prefill_w1_w3_prg_config(seq_len)
    pc_w2 = cfg.prefill_w2_prg_config(seq_len)

    a1 = ttnn.from_torch(
        torch.randn((1, 1, seq_len, dim), dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b1 = ttnn.from_torch(
        torch.randn((1, 1, dim, hidden_dim), dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b2 = ttnn.from_torch(
        torch.randn((1, 1, hidden_dim, dim), dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for _ in range(warmup_iters):
        y = ttnn.matmul(a1, b1, program_config=pc_w1)
        z = ttnn.matmul(y, b2, program_config=pc_w2)
        _ = ttnn.to_torch(z)

    samples_ms: list[float] = []
    for _ in range(measure_iters):
        start_ns = time.time_ns()
        y = ttnn.matmul(a1, b1, program_config=pc_w1)
        z = ttnn.matmul(y, b2, program_config=pc_w2)
        _ = ttnn.to_torch(z)
        samples_ms.append((time.time_ns() - start_ns) / 1_000_000.0)

    sorted_samples = sorted(samples_ms)
    return {
        "name": case["name"],
        "seq_len": seq_len,
        "dim": dim,
        "hidden_dim": hidden_dim,
        "pc_w1_grid": [
            int(pc_w1.compute_with_storage_grid_size.x),
            int(pc_w1.compute_with_storage_grid_size.y),
        ],
        "pc_w2_grid": [
            int(pc_w2.compute_with_storage_grid_size.x),
            int(pc_w2.compute_with_storage_grid_size.y),
        ],
        "perf_ms": {
            "mean": statistics.fmean(samples_ms),
            "p50": _percentile(sorted_samples, 0.50),
            "p95": _percentile(sorted_samples, 0.95),
            "samples": samples_ms,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MLP1D prefill benchmark on N150")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "/home/ubuntu/tt-metal/tests/sweep_framework/benchmark_protocol/generated/deepseek_oriented/mlp1d_prefill_n150_benchmark_last.json"
        ),
    )
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=10)
    args = parser.parse_args()

    cases = [
        {"name": "mid_2048_4096", "seq_len": 32, "dim": 2048, "hidden_dim": 4096},
        {"name": "mid_1536_3072", "seq_len": 32, "dim": 1536, "hidden_dim": 3072},
        {"name": "small_1024_2048", "seq_len": 32, "dim": 1024, "hidden_dim": 2048},
    ]

    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
    try:
        rows = [_run_single(c, device, args.warmup_iters, args.measure_iters) for c in cases]
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
            "overall_p50_ms": _percentile(p50s, 0.50),
            "overall_p95_ms": _percentile(p95s, 0.95),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote benchmark JSON -> {args.out_json}")
    print(f"overall_p50_ms: {payload['summary']['overall_p50_ms']}")
    print(f"overall_p95_ms: {payload['summary']['overall_p95_ms']}")
    for r in rows:
        print(
            f"- {r['name']}: w1_grid={r['pc_w1_grid']}, w2_grid={r['pc_w2_grid']}, "
            f"p50={r['perf_ms']['p50']:.3f}ms, p95={r['perf_ms']['p95']:.3f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
