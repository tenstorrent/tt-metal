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
from tracy.common import PROFILER_LOGS_DIR
from tracy.process_ops_logs import get_device_data_generate_report


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


def _parse_shape(shape_str: str) -> list[int]:
    shape_str = shape_str.strip()
    if not (shape_str.startswith("(") and shape_str.endswith(")")):
        raise ValueError(f"Unexpected shape format: {shape_str}")
    items = [x.strip() for x in shape_str[1:-1].split(",")]
    return [int(x) for x in items if x]


def _to_memcfg(memcfg_json: dict) -> ttnn.MemoryConfig:
    buffer_type = memcfg_json.get("data", {}).get("buffer_type", "DRAM")
    if buffer_type == "L1":
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def _read_device_kernel_duration_ns(device: ttnn.Device) -> float | None:
    ttnn.ReadDeviceProfiler(device)
    rows = get_device_data_generate_report(
        PROFILER_LOGS_DIR,
        outputFolder=None,
        date=None,
        nameAppend=None,
        export_csv=False,
        cleanup_device_log=True,
        device_analysis_types=("device_kernel_duration",),
    )
    vals: list[float] = []
    for row in rows:
        v = row.get("DEVICE KERNEL DURATION [ns]")
        if v in (None, "-"):
            continue
        vals.append(float(v))
    return sum(vals) if vals else None


def _run_case(
    case: dict,
    vector: dict,
    device: ttnn.Device,
    warmup_iters: int,
    measure_iters: int,
    measure_device_kernel: bool,
) -> dict:
    a_shape = _parse_shape(vector["input_a_shape"])
    b_shape = _parse_shape(vector["input_b_shape"])
    if len(a_shape) != 4 or len(b_shape) not in (2, 4):
        raise ValueError(f"Unsupported shapes for input_hash={case['input_hash']}: {a_shape}, {b_shape}")

    m = int(a_shape[-2])
    k = int(a_shape[-1])
    n = int(b_shape[-1])

    a = ttnn.from_torch(
        torch.randn(tuple(a_shape), dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_to_memcfg(vector["input_a_memory_config"]),
    )
    b = ttnn.from_torch(
        torch.randn(tuple(b_shape), dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_to_memcfg(vector["input_b_memory_config"]),
    )
    out_memcfg = _to_memcfg(vector["output_memory_config"])

    for _ in range(warmup_iters):
        out = ttnn.matmul(a, b, memory_config=out_memcfg)
        _ = ttnn.to_torch(out)
        if measure_device_kernel:
            _ = _read_device_kernel_duration_ns(device)

    e2e_samples_ms: list[float] = []
    kernel_samples_ns: list[float] = []
    for _ in range(measure_iters):
        start_ns = time.time_ns()
        out = ttnn.matmul(a, b, memory_config=out_memcfg)
        _ = ttnn.to_torch(out)
        e2e_samples_ms.append((time.time_ns() - start_ns) / 1_000_000.0)
        if measure_device_kernel:
            kernel_ns = _read_device_kernel_duration_ns(device)
            if kernel_ns is not None:
                kernel_samples_ns.append(kernel_ns)

    e2e_sorted = sorted(e2e_samples_ms)
    kernel_sorted = sorted(kernel_samples_ns)
    return {
        "name": case["name"],
        "regime": case["regime"],
        "input_hash": case["input_hash"],
        "source": str(vector.get("traced_source")),
        "shape_mkn": [m, k, n],
        "memory": {
            "input_a": vector["input_a_memory_config"]["data"]["buffer_type"],
            "input_b": vector["input_b_memory_config"]["data"]["buffer_type"],
            "output": vector["output_memory_config"]["data"]["buffer_type"],
        },
        "e2e_ms": {
            "mean": statistics.fmean(e2e_samples_ms),
            "p50": _percentile(e2e_sorted, 0.50),
            "p95": _percentile(e2e_sorted, 0.95),
            "samples": e2e_samples_ms,
        },
        "device_kernel_ns": {
            "available": bool(kernel_samples_ns),
            "mean": statistics.fmean(kernel_samples_ns) if kernel_samples_ns else None,
            "p50": _percentile(kernel_sorted, 0.50) if kernel_samples_ns else None,
            "p95": _percentile(kernel_sorted, 0.95) if kernel_samples_ns else None,
            "samples": kernel_samples_ns,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Kernel-focused N150 matmul microbench by traced-regime manifest")
    parser.add_argument(
        "--regimes-json",
        type=Path,
        default=Path("/home/ubuntu/tt-metal/tests/sweep_framework/benchmark_protocol/matmul_n150_regimes.json"),
    )
    parser.add_argument(
        "--all-vectors-json",
        type=Path,
        default=Path(
            "/home/ubuntu/tt-metal/tests/sweep_framework/benchmark_protocol/generated/matmul_n150_protocol_all.json"
        ),
    )
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=8)
    parser.add_argument("--measure-device-kernel", action="store_true")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "/home/ubuntu/tt-metal/tests/sweep_framework/benchmark_protocol/generated/deepseek_oriented/matmul_n150_kernelbench_last.json"
        ),
    )
    args = parser.parse_args()

    regimes = json.loads(args.regimes_json.read_text())
    all_vectors = json.loads(args.all_vectors_json.read_text())["model_traced"]
    cases = regimes["cases"]

    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
    try:
        rows = []
        for case in cases:
            input_hash = case["input_hash"]
            if input_hash not in all_vectors:
                raise KeyError(f"input_hash {input_hash} not found in {args.all_vectors_json}")
            rows.append(
                _run_case(
                    case=case,
                    vector=all_vectors[input_hash],
                    device=device,
                    warmup_iters=args.warmup_iters,
                    measure_iters=args.measure_iters,
                    measure_device_kernel=args.measure_device_kernel,
                )
            )
    finally:
        ttnn.close_device(device)

    e2e_p50s = sorted(r["e2e_ms"]["p50"] for r in rows if r["e2e_ms"]["p50"] is not None)
    e2e_p95s = sorted(r["e2e_ms"]["p95"] for r in rows if r["e2e_ms"]["p95"] is not None)
    kernel_p50s = sorted(r["device_kernel_ns"]["p50"] for r in rows if r["device_kernel_ns"]["p50"] is not None)
    kernel_p95s = sorted(r["device_kernel_ns"]["p95"] for r in rows if r["device_kernel_ns"]["p95"] is not None)

    payload = {
        "regimes_json": str(args.regimes_json),
        "all_vectors_json": str(args.all_vectors_json),
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "measure_device_kernel": bool(args.measure_device_kernel),
        "results": rows,
        "summary": {
            "case_count": len(rows),
            "overall_e2e_p50_ms": _percentile(e2e_p50s, 0.50),
            "overall_e2e_p95_ms": _percentile(e2e_p95s, 0.95),
            "overall_kernel_p50_ns": _percentile(kernel_p50s, 0.50) if kernel_p50s else None,
            "overall_kernel_p95_ns": _percentile(kernel_p95s, 0.95) if kernel_p95s else None,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote benchmark JSON -> {args.out_json}")
    print(f"overall_e2e_p50_ms: {payload['summary']['overall_e2e_p50_ms']}")
    print(f"overall_e2e_p95_ms: {payload['summary']['overall_e2e_p95_ms']}")
    if args.measure_device_kernel:
        print(f"overall_kernel_p50_ns: {payload['summary']['overall_kernel_p50_ns']}")
        print(f"overall_kernel_p95_ns: {payload['summary']['overall_kernel_p95_ns']}")

    for row in rows:
        kernel_p50 = row["device_kernel_ns"]["p50"]
        kernel_p50_str = "n/a" if kernel_p50 is None else f"{kernel_p50:.1f}ns"
        print(
            f"- {row['name']} [{row['regime']}] mkn={row['shape_mkn']}: "
            f"e2e_p50={row['e2e_ms']['p50']:.3f}ms e2e_p95={row['e2e_ms']['p95']:.3f}ms "
            f"kernel_p50={kernel_p50_str}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
