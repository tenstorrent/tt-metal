#!/usr/bin/env python3
"""
Add all-reduce bandwidth analysis to experiment results.

For each DDP experiment with gradient_sync timing, computes:
  - grad_sync_bw_GBs: achieved all-reduce bandwidth (GB/s)
  - grad_sync_bw_util: fraction of theoretical link bandwidth achieved

Usage:
    python add_allreduce_bw.py experiments/all_results.json
    python add_allreduce_bw.py results.json --topology ring --theoretical-bw 48
    python add_allreduce_bw.py results.json -o results_with_bw.json --topology linear
"""

import argparse
import json
import sys
from pathlib import Path

DTYPE_BYTES = 2  # BF16


def get_grad_sync_ms(entry: dict) -> float | None:
    """Get gradient_sync time. Prefers profiler, falls back to naive."""
    if entry.get("timings"):
        first_dev = next(iter(entry["timings"]))
        return entry["timings"][first_dev].get("average", {}).get("gradient_sync_ms")

    if entry.get("naive_timings"):
        return (
            entry["naive_timings"]["device_host"]
            .get("average", {})
            .get("gradient_sync_ms")
        )

    return None


def compute_allreduce_bw(
    tensor_bytes: float,
    n_devices: int,
    time_s: float,
    topology: str,
    num_links: int = 1,
) -> float:
    """Compute effective per-link bandwidth for all-reduce.

    Ring:   each device sends 2*(N-1)/N * M bytes total.
            BW = 2*(N-1)*M / (N*T)
    Linear: each device sends N*M bytes total (naive).
            BW = N*M / T
    """
    if time_s <= 0 or n_devices < 2:
        return 0.0

    # 2 - because AR = RS + AG
    total_bytes_moved = (
        2
        * tensor_bytes
        * ((n_devices - 1) / n_devices)
        / num_links
        / (2 if topology == "ring" else 1)
    )
    return total_bytes_moved / time_s


def main():
    parser = argparse.ArgumentParser(description="Add all-reduce bandwidth analysis")
    parser.add_argument("results_json", help="Path to results JSON")
    parser.add_argument("-o", "--output", help="Output JSON (default: <input>_bw.json)")
    parser.add_argument(
        "--topology",
        choices=["linear", "ring"],
        default="linear",
        help="All-reduce topology (default: linear)",
    )
    parser.add_argument(
        "--theoretical-bw",
        type=float,
        default=48.0,
        help="Theoretical per-link bandwidth in GB/s (default: 48)",
    )
    parser.add_argument(
        "--tp-mem-shard",
        type=float,
        default=0.88,
        help="Fraction of model params that are TP-sharded (default: 0.88)",
    )
    args = parser.parse_args()

    results_path = Path(args.results_json)
    output = (
        Path(args.output)
        if args.output
        else results_path.with_name(results_path.stem + "_bw" + results_path.suffix)
    )

    results = json.loads(results_path.read_text())

    ddp_entries = [e for e in results if e.get("experiment", {}).get("ddp", 1) > 1]
    print(f"Loaded {len(results)} experiments, {len(ddp_entries)} with DDP > 1")
    print(f"Topology: {args.topology}, theoretical BW: {args.theoretical_bw} GB/s\n")

    for entry in ddp_entries:
        name = entry["name"]
        exp = entry.get("experiment", {})
        ddp = exp.get("ddp", 1)
        tp = exp.get("tp", 1)
        num_params = entry.get("num_parameters")
        tp_mem_shard = args.tp_mem_shard

        grad_sync_ms = get_grad_sync_ms(entry)

        if grad_sync_ms is None or grad_sync_ms <= 0:
            print(f"  [{name}] no gradient_sync timing — skipped")
            continue

        if num_params is None:
            print(f"  [{name}] no num_parameters — skipped")
            continue

        # Per-chip gradient bytes = total_params * params_fraction * dtype
        params_fraction = (1 - tp_mem_shard) + tp_mem_shard / tp
        tensor_bytes = num_params * params_fraction * DTYPE_BYTES

        time_s = grad_sync_ms / 1000.0
        bw = compute_allreduce_bw(tensor_bytes, ddp, time_s, args.topology)
        bw_gbs = bw / 1e9
        bw_util = bw_gbs / args.theoretical_bw * 100 if args.theoretical_bw > 0 else 0

        entry["allreduce"] = {
            "topology": args.topology,
            "ddp": ddp,
            "tp": tp,
            "params_fraction": round(params_fraction, 4),
            "tensor_bytes": round(tensor_bytes),
            "tensor_mb": round(tensor_bytes / 1e6, 1),
            "grad_sync_ms": round(grad_sync_ms, 3),
            "bw_GBs": round(bw_gbs, 2),
            "theoretical_bw_GBs": args.theoretical_bw,
            "bw_util_perc": round(bw_util, 1),
        }

        print(
            f"  [{name}] DDP={ddp} TP={tp} "
            f"tensor={tensor_bytes/1e6:.1f}MB "
            f"sync={grad_sync_ms:.1f}ms "
            f"BW={bw_gbs:.1f}GB/s "
            f"util={bw_util:.1f}%"
        )

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    annotated = sum(1 for e in results if "allreduce" in e)
    print(f"\n{annotated}/{len(results)} experiments annotated → {output}")


if __name__ == "__main__":
    main()
