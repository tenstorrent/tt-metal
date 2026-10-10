# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate a complete operator inventory without accessing physical devices."""

import argparse
import csv
import json
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.recover_full_profile_export import digest, passed_xml
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.operator_profile import build_report


def run(args):
    for root in (args.capture, args.baseline):
        passed_xml(root / "hardware.xml")
    files = list((args.capture / "tracy").rglob("ops_perf_results*.csv"))
    if len(files) != 1:
        raise ValueError("Require one completed full-model operation export")
    files = dict(ops=files[0], profile=args.capture / "profile.json", baseline=args.baseline / "profile.json")
    hashes = {name: digest(path) for name, path in files.items()}
    with files["ops"].open(newline="") as stream:
        report = build_report(
            csv.DictReader(stream), json.loads(files["profile"].read_text()), json.loads(files["baseline"].read_text())
        )
    if hashes != {name: digest(path) for name, path in files.items()}:
        raise ValueError("Capture changed during analysis")
    report.update(
        source_sha256=hashes, capture=str(args.capture), baseline=str(args.baseline), physical_devices_accessed=False
    )
    args.output.mkdir()
    save(args.output / "analysis.json", report)
    lines = [
        "# Complete operator profile",
        "",
        report["scope"],
        "",
        f"Context {report['input_tokens']}, batch {report['batch']}, {report['distinct_operation_types']} operation types.",
        f"Unprofiled restored-cache step {report['unprofiled_step_ms']:.3f} ms; profiled {report['profiled_step_ms']:.3f} ms.",
        f"Whole-step profiler overhead {report['profiler_overhead_fraction']:.2%}; this is not HTTP/natural-prompt throughput.",
        "",
        report["accounting"],
        "",
        "| Operation | Calls, range | Median kernel sum, ms |",
        "|---|---:|---:|",
    ]
    for row in report["inventory"]:
        lines.append(
            f"| {row['operation']} | {row['calls_range'][0]}-{row['calls_range'][1]} | {row['median_kernel_ms']:.4f} |"
        )
    lines += [
        "",
        "## RISC time including waits",
        "",
        "These durations are not active compute or physical bandwidth utilization. Missing measurements stay blank.",
        "",
        "| Operation | Reader, ms | Writer, ms | Compute, ms |",
        "|---|---:|---:|---:|",
    ]
    for row in report["inventory"]:
        values = [row["risc_wait_inclusive"][risc]["median_sum_ms"] for risc in ("reader", "writer", "compute")]
        lines.append(
            f"| {row['operation']} | " + " | ".join("-" if value is None else f"{value:.4f}" for value in values) + " |"
        )
    lines += [
        "",
        "## Disjoint firmware timelines",
        "",
        "The following family durations add within each listed replay only.",
        "",
    ]
    families = sorted({f for row in report["longest_rank_timelines"] for f in row["families_ms"]})
    lines += ["| Family | Replay 0, ms | Replay 1, ms | Replay 2, ms |", "|---|---:|---:|---:|"]
    for name in families:
        values = [row["families_ms"].get(name, 0) for row in report["longest_rank_timelines"]]
        lines.append(f"| {name} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += [
        "",
        "## Matmul weight-byte estimates",
        "",
        report["bandwidth_assumptions"],
        "These are not physical DRAM counters or a calibrated compute roofline.",
        "",
        "| Projection | Stored K x N | Calls | Kernel sum, ms | Weight GB/s | Assumed peak fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["projections"]:
        lines.append(
            f"| {row['projection']} | {row['stored_kn'][0]} x {row['stored_kn'][1]} | {row['calls_per_step']} | "
            f"{row['median_kernel_ms']:.3f} | {row['encoded_weight_gbs']:.1f} | {row['fraction_of_assumed_peak']:.1%} |"
        )
    (args.output / "INVENTORY.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
