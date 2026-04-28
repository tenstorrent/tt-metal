#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark ttnn.rand latency (cache miss and cache hit) in a reproducible way.

This script is intended for before-vs-after comparisons between two commits/branches.
Run exactly the same command in both revisions and compare JSON/CSV outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import ttnn


LAYOUT_MAP = {
    "tile": ttnn.TILE_LAYOUT,
    "row-major": ttnn.ROW_MAJOR_LAYOUT,
    "row_major": ttnn.ROW_MAJOR_LAYOUT,
}

DTYPE_MAP = {
    "bfloat16": ttnn.bfloat16,
    "float32": ttnn.float32,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "int32": ttnn.int32,
}

MEMORY_CONFIG_MAP = {
    "dram": ttnn.DRAM_MEMORY_CONFIG,
    "l1": ttnn.L1_MEMORY_CONFIG,
}


@dataclass
class CaseResult:
    label: str
    shape: list[int]
    dtype: str
    layout: str
    memory_config: str
    mode: str
    mapper: str
    repeats: int
    warmup_iters: int
    hit_iters: int
    miss_samples_us: list[float]
    hit_samples_us: list[float]
    miss_median_us: Optional[float]
    miss_p95_us: Optional[float]
    hit_median_us: Optional[float]
    hit_p95_us: Optional[float]
    cache_entries_after_case: Optional[int]


def parse_csv_ints(text: str, name: str) -> list[int]:
    try:
        values = [int(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: '{text}'") from exc
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return values


def parse_shapes(shape_args: list[str]) -> list[list[int]]:
    shapes: list[list[int]] = []
    for s in shape_args:
        shape = parse_csv_ints(s, "shape")
        if any(dim <= 0 for dim in shape):
            raise ValueError(f"All shape dimensions must be > 0: {shape}")
        shapes.append(shape)
    return shapes


def percentile(sorted_values: list[float], p: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.median(values)


def p95(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return percentile(sorted(values), 0.95)


def ns_to_us(ns: int) -> float:
    return ns / 1000.0


def maybe_enable_program_cache(device: Any, enable: bool) -> None:
    if not enable:
        return
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()


def maybe_disable_and_clear_program_cache(device: Any) -> None:
    if hasattr(device, "disable_and_clear_program_cache"):
        device.disable_and_clear_program_cache()
    elif hasattr(device, "clear_program_cache"):
        device.clear_program_cache()


def maybe_clear_program_cache(device: Any) -> None:
    if hasattr(device, "clear_program_cache"):
        device.clear_program_cache()


def maybe_num_program_cache_entries(device: Any) -> Optional[int]:
    if hasattr(device, "num_program_cache_entries"):
        try:
            return int(device.num_program_cache_entries())
        except Exception:
            return None
    return None


def build_mesh_mapper(mesh_shape: list[int], mapper: str, shard_dim: int) -> Optional[Any]:
    if mapper == "none":
        return None
    if mapper == "replicate":
        placements = [ttnn.PlacementReplicate() for _ in mesh_shape]
        return ttnn.MeshMapperConfig(placements)
    if mapper == "shard":
        if shard_dim < 0:
            raise ValueError("--shard-dim must be >= 0 for --mapper shard")
        placements = []
        for size in mesh_shape:
            if size > 1:
                placements.append(ttnn.PlacementShard(shard_dim))
            else:
                placements.append(ttnn.PlacementReplicate())
        return ttnn.MeshMapperConfig(placements)
    raise ValueError(f"Unsupported mapper mode: {mapper}")


def run_rand_once(
    *,
    mode: str,
    shape: list[int],
    device: Any,
    dtype_name: str,
    layout_name: str,
    memory_config_name: str,
    low: float,
    high: float,
    seed: int,
    mesh_mapper: Optional[Any],
) -> int:
    dtype = DTYPE_MAP[dtype_name]
    layout = LAYOUT_MAP[layout_name]
    memory_config = MEMORY_CONFIG_MAP[memory_config_name]
    start_ns = time.perf_counter_ns()
    if mode == "single":
        _ = ttnn.rand(
            shape,
            device=device,
            dtype=dtype,
            layout=layout,
            memory_config=memory_config,
            low=low,
            high=high,
            seed=seed,
        )
    else:
        kwargs = {
            "dtype": dtype,
            "layout": layout,
            "memory_config": memory_config,
            "low": low,
            "high": high,
            "seed": seed,
        }
        if mesh_mapper is not None:
            kwargs["mesh_mapper"] = mesh_mapper
        _ = ttnn.rand(shape, device, **kwargs)
    ttnn.synchronize_device(device)
    end_ns = time.perf_counter_ns()
    return end_ns - start_ns


def open_target_device(args: argparse.Namespace) -> tuple[Any, Optional[list[int]]]:
    if args.mode == "single":
        device = ttnn.open_device(device_id=args.device_id)
        return device, None

    mesh_shape = parse_csv_ints(args.mesh_shape, "mesh-shape")
    if len(mesh_shape) > 3:
        raise ValueError("--mesh-shape supports up to 3 dimensions")

    mesh_shape_obj = ttnn.MeshShape(*mesh_shape)
    if args.mesh_device_ids:
        mesh_device_ids = parse_csv_ints(args.mesh_device_ids, "mesh-device-ids")
        device = ttnn.open_mesh_device(mesh_shape=mesh_shape_obj, device_ids=mesh_device_ids)
    else:
        device = ttnn.open_mesh_device(mesh_shape=mesh_shape_obj)
    return device, mesh_shape


def close_target_device(mode: str, device: Any) -> None:
    if mode == "single":
        ttnn.close_device(device)
    else:
        ttnn.close_mesh_device(device)


def benchmark_case(
    args: argparse.Namespace,
    *,
    case_label: str,
    shape: list[int],
    device: Any,
    mesh_shape: Optional[list[int]],
) -> CaseResult:
    mesh_mapper = None
    if args.mode == "mesh":
        mesh_mapper = build_mesh_mapper(mesh_shape or [], args.mapper, args.shard_dim)

    miss_samples: list[float] = []
    hit_samples: list[float] = []

    for repeat_idx in range(args.repeats):
        if args.clear_cache_each_repeat:
            maybe_clear_program_cache(device)

        if args.measure_miss:
            miss_ns = run_rand_once(
                mode=args.mode,
                shape=shape,
                device=device,
                dtype_name=args.dtype,
                layout_name=args.layout,
                memory_config_name=args.memory_config,
                low=args.low,
                high=args.high,
                seed=args.seed,
                mesh_mapper=mesh_mapper,
            )
            miss_samples.append(ns_to_us(miss_ns))

        for _ in range(args.warmup_iters):
            _ = run_rand_once(
                mode=args.mode,
                shape=shape,
                device=device,
                dtype_name=args.dtype,
                layout_name=args.layout,
                memory_config_name=args.memory_config,
                low=args.low,
                high=args.high,
                seed=args.seed,
                mesh_mapper=mesh_mapper,
            )

        for _ in range(args.hit_iters):
            hit_ns = run_rand_once(
                mode=args.mode,
                shape=shape,
                device=device,
                dtype_name=args.dtype,
                layout_name=args.layout,
                memory_config_name=args.memory_config,
                low=args.low,
                high=args.high,
                seed=args.seed,
                mesh_mapper=mesh_mapper,
            )
            hit_samples.append(ns_to_us(hit_ns))

        if args.sleep_ms_between_repeats > 0 and repeat_idx + 1 < args.repeats:
            time.sleep(args.sleep_ms_between_repeats / 1000.0)

    return CaseResult(
        label=case_label,
        shape=shape,
        dtype=args.dtype,
        layout=args.layout,
        memory_config=args.memory_config,
        mode=args.mode,
        mapper=args.mapper if args.mode == "mesh" else "none",
        repeats=args.repeats,
        warmup_iters=args.warmup_iters,
        hit_iters=args.hit_iters,
        miss_samples_us=miss_samples if args.emit_raw_samples else [],
        hit_samples_us=hit_samples if args.emit_raw_samples else [],
        miss_median_us=median(miss_samples),
        miss_p95_us=p95(miss_samples),
        hit_median_us=median(hit_samples),
        hit_p95_us=p95(hit_samples),
        cache_entries_after_case=maybe_num_program_cache_entries(device),
    )


def write_csv(path: Path, results: list[CaseResult], include_raw: bool) -> None:
    fieldnames = [
        "label",
        "shape",
        "dtype",
        "layout",
        "memory_config",
        "mode",
        "mapper",
        "repeats",
        "warmup_iters",
        "hit_iters",
        "miss_median_us",
        "miss_p95_us",
        "hit_median_us",
        "hit_p95_us",
        "cache_entries_after_case",
    ]
    if include_raw:
        fieldnames.extend(["miss_samples_us", "hit_samples_us"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "label": result.label,
                "shape": "x".join(str(x) for x in result.shape),
                "dtype": result.dtype,
                "layout": result.layout,
                "memory_config": result.memory_config,
                "mode": result.mode,
                "mapper": result.mapper,
                "repeats": result.repeats,
                "warmup_iters": result.warmup_iters,
                "hit_iters": result.hit_iters,
                "miss_median_us": result.miss_median_us,
                "miss_p95_us": result.miss_p95_us,
                "hit_median_us": result.hit_median_us,
                "hit_p95_us": result.hit_p95_us,
                "cache_entries_after_case": result.cache_entries_after_case,
            }
            if include_raw:
                row["miss_samples_us"] = json.dumps(result.miss_samples_us)
                row["hit_samples_us"] = json.dumps(result.hit_samples_us)
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    epilog = """
Examples:

  # Single device, two shapes, write JSON:
  python tests/scripts/benchmark_rand_perf.py \\
    --mode single --device-id 0 \\
    --shape 1024,1024 --shape 4096,4096 \\
    --dtype float32 --layout tile --memory-config dram \\
    --low 0.0 --high 1.0 --seed 42 \\
    --measure-miss --warmup-iters 5 --hit-iters 100 --repeats 3 \\
    --output before.json --output-format json --label before

  # Mesh (1x2), sharded mapper:
  python tests/scripts/benchmark_rand_perf.py \\
    --mode mesh --mesh-shape 1,2 --mapper shard --shard-dim 0 \\
    --shape 2048,1024 --dtype float32 --layout tile --memory-config dram \\
    --measure-miss --warmup-iters 5 --hit-iters 100 --repeats 3 \\
    --output after.json --output-format json --label after

Parameter notes:
  --shape: Comma-separated dimensions. Repeat --shape to benchmark multiple shapes.
  --mode: single | mesh.
  --mapper (mesh only): none | replicate | shard.
  --mesh-device-ids: Optional comma-separated device id list, e.g. 0,1,2,3.
  --output-format: json | csv | both.
  --emit-raw-samples: include per-iteration samples in output.
  --clear-cache-each-repeat: clear cache before each repeat (default enabled).
"""
    parser = argparse.ArgumentParser(
        description="Benchmark ttnn.rand cache-miss and cache-hit latency.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument("--mode", choices=["single", "mesh"], default="single", help="Execution mode.")
    parser.add_argument("--device-id", type=int, default=0, help="Single-device id (used when --mode single).")
    parser.add_argument(
        "--mesh-shape",
        type=str,
        default="1,2",
        help="Mesh shape for mesh mode, comma-separated dimensions (e.g. 1,2 or 2,2).",
    )
    parser.add_argument(
        "--mesh-device-ids",
        type=str,
        default="",
        help="Optional comma-separated device ids for mesh mode (e.g. 0,1,2,3).",
    )
    parser.add_argument(
        "--mapper",
        choices=["none", "replicate", "shard"],
        default="none",
        help="Mesh mapper type (mesh mode only).",
    )
    parser.add_argument(
        "--shard-dim",
        type=int,
        default=0,
        help="Tensor dimension to shard when --mapper shard (mesh mode only).",
    )

    parser.add_argument(
        "--shape",
        action="append",
        required=True,
        help="Shape as comma-separated dims; repeat this flag for multiple cases.",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP.keys()), default="float32", help="ttnn dtype.")
    parser.add_argument("--layout", choices=["tile", "row-major", "row_major"], default="tile", help="Tensor layout.")
    parser.add_argument("--memory-config", choices=["dram", "l1"], default="dram", help="Output memory config.")
    parser.add_argument("--low", type=float, default=0.0, help="Random lower bound.")
    parser.add_argument("--high", type=float, default=1.0, help="Random upper bound (exclusive upper bound in kernel).")
    parser.add_argument("--seed", type=int, default=42, help="Seed passed to ttnn.rand.")

    parser.add_argument(
        "--measure-miss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure first-call cache miss latency per repeat.",
    )
    parser.add_argument("--warmup-iters", type=int, default=5, help="Warmup iterations before hit timing.")
    parser.add_argument("--hit-iters", type=int, default=100, help="Measured cache-hit iterations per repeat.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per shape.")
    parser.add_argument(
        "--sleep-ms-between-repeats",
        type=int,
        default=0,
        help="Optional sleep between repeats in milliseconds.",
    )

    parser.add_argument(
        "--use-program-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable program cache on target device.",
    )
    parser.add_argument(
        "--clear-cache-each-repeat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear program cache before each repeat.",
    )

    parser.add_argument("--label", type=str, default="", help="Optional run label (e.g. before, after).")
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "both"],
        default="json",
        help="Output format for results.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output file path. For --output-format both, this is used as base path stem.",
    )
    parser.add_argument(
        "--emit-raw-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include per-iteration samples in output payload.",
    )
    return parser


def default_output_paths(args: argparse.Namespace) -> tuple[Optional[Path], Optional[Path]]:
    timestamp = int(time.time())
    label_part = f"_{args.label}" if args.label else ""
    if args.output:
        out = Path(args.output)
        if args.output_format == "json":
            return out, None
        if args.output_format == "csv":
            return None, out
        return out.with_suffix(".json"), out.with_suffix(".csv")

    base = Path(f"rand_perf{label_part}_{timestamp}")
    if args.output_format == "json":
        return base.with_suffix(".json"), None
    if args.output_format == "csv":
        return None, base.with_suffix(".csv")
    return base.with_suffix(".json"), base.with_suffix(".csv")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.low >= args.high:
        raise ValueError("--low must be < --high")
    if args.warmup_iters < 0 or args.hit_iters < 0 or args.repeats <= 0:
        raise ValueError("--warmup-iters and --hit-iters must be >= 0; --repeats must be > 0")
    if args.mode == "single" and args.mapper != "none":
        raise ValueError("--mapper is only valid in mesh mode")

    shapes = parse_shapes(args.shape)
    device, mesh_shape = open_target_device(args)
    maybe_enable_program_cache(device, args.use_program_cache)

    results: list[CaseResult] = []
    try:
        for shape in shapes:
            case_label = f"{args.label}:" if args.label else ""
            case_label += f"{args.mode}:{'x'.join(map(str, shape))}:{args.dtype}:{args.layout}:{args.memory_config}"
            result = benchmark_case(args, case_label=case_label, shape=shape, device=device, mesh_shape=mesh_shape)
            results.append(result)
    finally:
        maybe_disable_and_clear_program_cache(device)
        close_target_device(args.mode, device)

    payload = {
        "script": "benchmark_rand_perf.py",
        "label": args.label,
        "mode": args.mode,
        "mapper": args.mapper if args.mode == "mesh" else "none",
        "parameters": {
            "dtype": args.dtype,
            "layout": args.layout,
            "memory_config": args.memory_config,
            "low": args.low,
            "high": args.high,
            "seed": args.seed,
            "measure_miss": args.measure_miss,
            "warmup_iters": args.warmup_iters,
            "hit_iters": args.hit_iters,
            "repeats": args.repeats,
            "use_program_cache": args.use_program_cache,
            "clear_cache_each_repeat": args.clear_cache_each_repeat,
            "emit_raw_samples": args.emit_raw_samples,
        },
        "results": [asdict(r) for r in results],
    }

    json_path, csv_path = default_output_paths(args)
    if json_path is not None:
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[rand-perf] wrote JSON: {json_path}")
    if csv_path is not None:
        write_csv(csv_path, results, include_raw=args.emit_raw_samples)
        print(f"[rand-perf] wrote CSV: {csv_path}")

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[rand-perf] ERROR: {e}", file=sys.stderr)
        raise
