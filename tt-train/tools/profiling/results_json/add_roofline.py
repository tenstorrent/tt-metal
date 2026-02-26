#!/usr/bin/env python3
"""
Add roofline analysis to experiment results.

For each TP=1 experiment, runs the roofline model to get theoretical
minimum times and FLOP counts, then computes:
  - roofline_perc: what fraction of roofline is achieved (higher = better)
  - mfu_perc: model FLOP utilization vs peak BF16 (160 TFLOPs/s)

Usage:
    python add_roofline.py experiments/all_results.json /path/to/tt-train-roofline
    python add_roofline.py results.json /path/to/tt-train-roofline -o results_with_roofline.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_BF16_PEAK_TFLOPS = 160.0

TIMING_RE = re.compile(
    r"(Forward|Backward|Optimizer|Grad Clip|Total):\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+TFLOPs\)"
)


def find_training_config(entry: dict) -> str | None:
    """Find the training_config.yaml path for an experiment."""
    cmd = entry.get("command", "")
    m = re.search(r"--config\s+(\S+)", cmd)
    if m:
        path = m.group(1).strip('"').strip("'")
        if os.path.exists(path):
            return path
    return None


def run_roofline(
    roofline_dir: Path, config_path: str, batch: int, hardware: str = "p100"
) -> dict | None:
    """Run roofline tool and parse output."""
    cmd = [
        sys.executable,
        "-m",
        "roofline.examples.training",
        "--config",
        config_path,
        "--batch",
        str(batch),
        "--hardware",
        hardware,
        "--no-plot",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(roofline_dir), timeout=120
        )
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None

    parsed = {}
    for m in TIMING_RE.finditer(result.stdout):
        phase = m.group(1).lower().replace(" ", "_")
        parsed[phase] = {
            "roofline_ms": float(m.group(2)),
            "flops_tflops": float(m.group(3)),
        }

    return parsed if parsed else None


def get_measured_phase_ms(entry: dict, phase: str) -> float | None:
    """Get measured time for a phase. Prefers profiler, falls back to naive."""
    if entry.get("timings"):
        first_dev = next(iter(entry["timings"]))
        avg = entry["timings"][first_dev].get("average", {})
        return avg.get(phase)

    if entry.get("naive_timings"):
        avg = entry["naive_timings"]["device_host"].get("average", {})
        return avg.get(phase)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Add roofline analysis to experiment results"
    )
    parser.add_argument("results_json", help="Path to extracted_results JSON")
    parser.add_argument("roofline_dir", help="Path to tt-train-roofline directory")
    parser.add_argument(
        "-o", "--output", help="Output JSON (default: <input>_roofline.json)"
    )
    parser.add_argument(
        "--hardware",
        default="p100",
        help="Hardware config for roofline tool (default: p100)",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=DEFAULT_BF16_PEAK_TFLOPS,
        help=f"Peak BF16 TFLOPs/s for MFU calculation (default: {DEFAULT_BF16_PEAK_TFLOPS})",
    )
    args = parser.parse_args()

    results_path = Path(args.results_json)
    roofline_dir = Path(args.roofline_dir)
    output = (
        Path(args.output)
        if args.output
        else results_path.with_name(
            results_path.stem + "_roofline" + results_path.suffix
        )
    )

    if not roofline_dir.exists():
        print(f"Roofline directory not found: {roofline_dir}")
        sys.exit(1)

    results = json.loads(results_path.read_text())

    tp1_entries = [e for e in results if e.get("experiment", {}).get("tp", 1) == 1]
    print(f"Loaded {len(results)} experiments, {len(tp1_entries)} with TP=1\n")

    for entry in tp1_entries:
        name = entry["name"]
        exp = entry.get("experiment", {})
        local_batch = exp.get("local_batch", 1)
        uses_checkpointing = exp.get("runner_type") == "memory_efficient"

        config_path = find_training_config(entry)
        if not config_path:
            print(f"  [{name}] no training config found — skipped")
            continue

        print(f"  [{name}] batch={local_batch} ckpt={uses_checkpointing} ...", end=" ")

        roofline = run_roofline(roofline_dir, config_path, local_batch, args.hardware)
        if not roofline:
            print("roofline FAILED")
            continue

        rf_fwd = roofline.get("forward", {})
        rf_bwd = roofline.get("backward", {})
        rf_opt = roofline.get("optimizer", {})

        if uses_checkpointing:
            # Checkpointing: backward includes a full forward recomputation
            rf_bwd = {
                "roofline_ms": rf_bwd.get("roofline_ms", 0)
                + rf_fwd.get("roofline_ms", 0),
                "flops_tflops": rf_bwd.get("flops_tflops", 0)
                + rf_fwd.get("flops_tflops", 0),
            }

        rf_total_ms = (
            rf_fwd.get("roofline_ms", 0)
            + rf_bwd.get("roofline_ms", 0)
            + rf_opt.get("roofline_ms", 0)
        )
        rf_total_flops = (
            rf_fwd.get("flops_tflops", 0)
            + rf_bwd.get("flops_tflops", 0)
            + rf_opt.get("flops_tflops", 0)
        )

        # Measured times
        meas_fwd = get_measured_phase_ms(entry, "forward_ms")
        meas_bwd = get_measured_phase_ms(entry, "backward_ms")
        meas_opt = get_measured_phase_ms(entry, "optimizer_ms")

        roofline_data = {"checkpointing_adjusted": uses_checkpointing}

        for label, rf, meas_ms in [
            ("forward", rf_fwd, meas_fwd),
            ("backward", rf_bwd, meas_bwd),
            ("optimizer", rf_opt, meas_opt),
        ]:
            r_ms = rf.get("roofline_ms", 0)
            r_flops = rf.get("flops_tflops", 0)

            phase_data = {
                "roofline_ms": round(r_ms, 4),
                "flops_tflops": round(r_flops, 4),
            }

            if meas_ms and meas_ms > 0:
                phase_data["measured_ms"] = round(meas_ms, 3)
                phase_data["roofline_perc"] = round(r_ms / meas_ms * 100, 1)
                phase_data["mfu_perc"] = round(
                    r_flops / (meas_ms / 1000) / args.peak_tflops * 100, 1
                )

            roofline_data[label] = phase_data

        # Total — use full step time (includes gradient_sync, host overhead)
        meas_total = None
        throughput = entry.get("throughput", {})
        if throughput.get("step_time_ms"):
            meas_total = throughput["step_time_ms"]
        elif get_measured_phase_ms(entry, "total_ms"):
            meas_total = get_measured_phase_ms(entry, "total_ms")

        total_data = {
            "roofline_ms": round(rf_total_ms, 4),
            "flops_tflops": round(rf_total_flops, 4),
        }
        if meas_total and meas_total > 0:
            total_data["measured_ms"] = round(meas_total, 3)
            total_data["roofline_perc"] = round(rf_total_ms / meas_total * 100, 1)
            total_data["mfu_perc"] = round(
                rf_total_flops / (meas_total / 1000) / args.peak_tflops * 100, 1
            )

        roofline_data["total"] = total_data
        entry["roofline"] = roofline_data

        # Print summary
        parts = []
        for lbl in ["forward", "backward", "optimizer", "total"]:
            rd = roofline_data.get(lbl, {})
            if "roofline_perc" in rd:
                parts.append(f"{lbl[:3]}={rd['roofline_perc']:.0f}%")
        mfu_total = roofline_data.get("total", {}).get("mfu_perc")
        mfu_str = f" MFU={mfu_total:.1f}%" if mfu_total else ""
        print(f"roofline: {' '.join(parts)}{mfu_str}")

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    annotated = sum(1 for e in results if "roofline" in e)
    print(f"\n{annotated}/{len(results)} experiments annotated → {output}")


if __name__ == "__main__":
    main()
