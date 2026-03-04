#!/usr/bin/env python3
"""Convert all_results_1.json to a CSV file with one row per experiment."""

import csv
import json
import re
import sys
from pathlib import Path

COLUMNS = [
    "name",
    "batch_size",
    "n_blocks",
    "tp",
    "dp",
    "grad_acc",
    "dram_peak_mb",
    "runner_type",
    "fwd_ms",
    "bwd_ms",
    "opt_ms",
    "grad_sync_ms",
    "step_time_ms",
    "profiler",
    "fwd_roofline_ms",
    "bwd_roofline_ms",
    "opt_roofline_ms",
    "total_roofline_ms",
    "fwd_roofline_perc",
    "bwd_roofline_perc",
    "opt_roofline_perc",
    "total_roofline_perc",
    "fwd_mfu_perc",
    "bwd_mfu_perc",
    "opt_mfu_perc",
    "total_mfu_perc",
    "grad_sync_bw_GBs",
    "grad_sync_util_perc",
    "fwd_ccl_util_perc",
    "bwd_ccl_util_perc",
    "total_ccl_util_perc",
    "t/s",
    "t/s/d",
    "fwd_ccl_ms",
    "bwd_ccl_ms",
    "opt_ccl_ms",
    "total_ccl_ms",
]


def _parse_name(name: str, field: str) -> int | None:
    """Extract integer value for a field token from the experiment name (e.g. tp4 -> 4)."""
    m = re.search(rf"(?:^|_){re.escape(field)}(\d+)(?:_|$)", name)
    return int(m.group(1)) if m else None


def _avg_timings(entry: dict) -> dict:
    """Return the averaged timing dict from tracy timings or naive_timings, or empty dict."""
    timings = entry.get("timings") or {}
    if timings:
        device_key = "device_0" if "device_0" in timings else next(iter(timings), None)
        if device_key:
            return timings[device_key].get("average") or {}

    # Fall back to naive_timings
    naive = entry.get("naive_timings") or {}
    if naive:
        device_key = (
            "device_host" if "device_host" in naive else next(iter(naive), None)
        )
        if device_key:
            return naive[device_key].get("average") or {}

    return {}


def _step_time(entry: dict, avg: dict) -> float | None:
    """Return step time in ms: total_ms from timings, falling back to total_step_ms."""
    if "total_ms" in avg:
        return avg["total_ms"]
    step_timings = entry.get("step_timings") or {}
    step_times = step_timings.get("step_times") or {}
    step_avg = step_times.get("average") or {}
    return step_avg.get("total_step_ms")


def _profiler_label(entry: dict) -> str:
    if entry.get("naive_timings") is not None:
        return "naive"
    if entry.get("profiler"):
        return "tracy"
    return "none"


def _ccl_fields(avg: dict) -> dict:
    """Compute CCL (collective communication) overhead columns from rs+ag pairs."""
    if not any(k in avg for k in ("fwd_rs_ms", "fwd_ag_ms", "bwd_rs_ms", "bwd_ag_ms")):
        return {
            "fwd_ccl_ms": None,
            "bwd_ccl_ms": None,
            "opt_ccl_ms": None,
            "total_ccl_ms": None,
        }
    fwd = (avg.get("fwd_rs_ms") or 0) + (avg.get("fwd_ag_ms") or 0)
    bwd = (avg.get("bwd_rs_ms") or 0) + (avg.get("bwd_ag_ms") or 0)
    opt = (avg.get("opt_rs_ms") or 0) + (avg.get("opt_ag_ms") or 0)
    grad_sync = avg.get("gradient_sync_ms") or 0
    return {
        "fwd_ccl_ms": fwd,
        "bwd_ccl_ms": bwd,
        "opt_ccl_ms": opt,
        "total_ccl_ms": fwd + bwd + opt + grad_sync,
    }


def process_entry(entry: dict) -> dict:
    name = entry["name"]
    exp = entry.get("experiment") or {}
    memory = entry.get("memory") or {}
    avg = _avg_timings(entry)
    roofline = entry.get("roofline") or {}

    tp = exp.get("tp") or _parse_name(name, "tp") or 1
    dp = exp.get("ddp") or _parse_name(name, "ddp") or 1
    grad_acc = exp.get("grad_acc") or _parse_name(name, "ga") or 1

    step_time = _step_time(entry, avg)

    throughput = entry.get("throughput") or {}
    fwd_r = roofline.get("forward") or {}
    bwd_r = roofline.get("backward") or {}
    opt_r = roofline.get("optimizer") or {}
    tot_r = roofline.get("total") or {}
    rf_ccl = roofline.get("ccl") or {}

    return {
        "name": name,
        "batch_size": exp.get("local_batch"),
        "n_blocks": exp.get("num_blocks"),
        "tp": tp,
        "dp": dp,
        "grad_acc": grad_acc,
        "dram_peak_mb": memory.get("overall_dram_peak_mb"),
        "runner_type": exp.get("runner_type"),
        "fwd_ms": avg.get("forward_ms"),
        "bwd_ms": avg.get("backward_ms"),
        "opt_ms": avg.get("optimizer_ms"),
        "grad_sync_ms": avg.get("gradient_sync_ms"),
        "step_time_ms": step_time,
        "profiler": _profiler_label(entry),
        "fwd_roofline_ms": fwd_r.get("roofline_ms"),
        "bwd_roofline_ms": bwd_r.get("roofline_ms"),
        "opt_roofline_ms": opt_r.get("roofline_ms"),
        "total_roofline_ms": tot_r.get("roofline_ms"),
        "fwd_roofline_perc": fwd_r.get("roofline_perc"),
        "bwd_roofline_perc": bwd_r.get("roofline_perc"),
        "opt_roofline_perc": opt_r.get("roofline_perc"),
        "total_roofline_perc": tot_r.get("roofline_perc"),
        "fwd_mfu_perc": fwd_r.get("mfu_perc"),
        "bwd_mfu_perc": bwd_r.get("mfu_perc"),
        "opt_mfu_perc": opt_r.get("mfu_perc"),
        "total_mfu_perc": tot_r.get("mfu_perc"),
        "grad_sync_bw_GBs": rf_ccl.get("grad_sync_bw_GBs"),
        "grad_sync_util_perc": rf_ccl.get("grad_sync_util_perc"),
        "fwd_ccl_util_perc": rf_ccl.get("fwd_ccl_util_perc"),
        "bwd_ccl_util_perc": rf_ccl.get("bwd_ccl_util_perc"),
        "total_ccl_util_perc": rf_ccl.get("total_ccl_util_perc"),
        "t/s": throughput.get("tokens_per_sec"),
        "t/s/d": throughput.get("tokens_per_sec_per_device"),
        **_ccl_fields(avg),
    }


def to_dataframe(data: list[dict]) -> "pd.DataFrame":
    """Convert results JSON list to a pandas DataFrame."""
    import pandas as _pd

    rows = [
        {
            k: round(v, 2) if isinstance(v, float) else v
            for k, v in process_entry(e).items()
        }
        for e in data
    ]
    return _pd.DataFrame(rows, columns=COLUMNS)


def main():
    if len(sys.argv) < 2:
        default = Path(__file__).parent / "experiments" / "all_results_1.json"
        input_path = default
    else:
        input_path = Path(sys.argv[1])

    output_path = (
        input_path.with_suffix(".csv") if len(sys.argv) < 3 else Path(sys.argv[2])
    )

    with open(input_path) as f:
        data = json.load(f)

    rows = [
        {
            k: round(v, 2) if isinstance(v, float) else v
            for k, v in process_entry(e).items()
        }
        for e in data
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
