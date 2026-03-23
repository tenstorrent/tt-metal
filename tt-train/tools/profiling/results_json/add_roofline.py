# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Add roofline analysis to experiment results.

For each experiment, runs the roofline model to get theoretical minimum
times and FLOP counts, then computes:
  - roofline_perc: what fraction of roofline is achieved (higher = better)
  - mfu_perc: model FLOP utilization vs peak BF16
  - grad_sync_bw_GBs / grad_sync_util_perc: DDP all-reduce bandwidth & utilization
  - fwd/bwd/total_ccl_util_perc: TP CCL utilization (theoretical / achieved)

Usage:
    python add_roofline.py experiments/all_results.json /path/to/tt-train-roofline
    python add_roofline.py results.json /path/to/tt-train-roofline -o results_with_roofline.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_BF16_PEAK_TFLOPS = 160.0

_run_model_roofline = None


def _init_roofline(roofline_dir: str | Path):
    """Add roofline_dir to sys.path and import the roofline tool."""
    global _run_model_roofline
    if _run_model_roofline is not None:
        return

    roofline_dir = str(Path(roofline_dir).resolve())
    if roofline_dir not in sys.path:
        sys.path.insert(0, roofline_dir)

    from roofline.examples.training import run_model_roofline

    _run_model_roofline = run_model_roofline


def find_training_config(entry: dict) -> str | None:
    """Find the training_config.yaml path for an experiment."""
    cmd = entry.get("command", "")
    m = re.search(r"--config\s+(\S+)", cmd)
    if m:
        path = m.group(1).strip('"').strip("'")
        if os.path.exists(path):
            return path
    return None


def run_roofline(config_path: str, hardware: str = "p100") -> dict | None:
    """Run roofline tool programmatically and return the result dictionary.

    All model, batch, and parallelism settings are read from the training
    config YAML so the roofline tool sees the full picture (including CCL).
    """
    try:
        result = _run_model_roofline(
            config=config_path,
            hardware=hardware,
            plot_memory=False,
            verbose=False,
        )
    except Exception as e:
        print(f"  Roofline failed for {config_path}: {e}")
        return None

    if not result or "timing_breakdown" not in result:
        err = result.get("error", "") if isinstance(result, dict) else ""
        print(f"  Roofline returned no timing_breakdown for {config_path}: {err}")
        return None

    return result


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


def enrich_roofline(
    results: list[dict],
    roofline_dir: str | Path,
    hardware: str = "bh_glx",
    peak_tflops: float | None = None,
) -> int:
    """Add roofline data to all experiments in-place. Returns count of annotated entries."""
    _init_roofline(roofline_dir)

    for entry in results:
        _enrich_one(entry, hardware, peak_tflops)

    return sum(1 for e in results if "roofline" in e)


def _extract_phase(timing_breakdown: dict, phase: str) -> dict:
    """Extract roofline_ms and flops_tflops from a timing_breakdown phase entry."""
    phase_data = timing_breakdown.get(phase, {})
    return {
        "roofline_ms": phase_data.get("time_ms", 0),
        "flops_tflops": phase_data.get("tflops", 0),
    }


def _get_measured_tp_ccl_ms(entry: dict, phase: str) -> float | None:
    """Get measured TP CCL time (rs + ag) for a phase.

    phase should be "fwd", "bwd", or "total".
    Only available from profiler timings (not naive).
    """
    if not entry.get("timings"):
        return None
    first_dev = next(iter(entry["timings"]))
    avg = entry["timings"][first_dev].get("average", {})

    if phase == "total":
        fwd = _get_measured_tp_ccl_ms(entry, "fwd")
        bwd = _get_measured_tp_ccl_ms(entry, "bwd")
        if fwd is not None and bwd is not None:
            return fwd + bwd
        return None

    rs = avg.get(f"{phase}_rs_ms")
    ag = avg.get(f"{phase}_ag_ms")
    if rs is not None and ag is not None:
        return rs + ag
    return None


def _enrich_one(entry, hardware, peak_tflops_override):
    exp = entry.get("experiment", {})
    uses_checkpointing = exp.get("runner_type") == "memory_efficient"

    config_path = find_training_config(entry)
    if not config_path:
        return

    result = run_roofline(config_path, hardware)
    if not result:
        return

    timing = result["timing_breakdown"]
    hw = result.get("hardware", {})
    ccl = result.get("ccl", {})
    peak_tflops = peak_tflops_override or hw.get("peak_tflops_hifi4", DEFAULT_BF16_PEAK_TFLOPS)

    rf_fwd = _extract_phase(timing, "forward")
    rf_bwd = _extract_phase(timing, "backward")
    rf_opt = _extract_phase(timing, "optimizer")

    if uses_checkpointing:
        rf_bwd = {
            "roofline_ms": rf_bwd["roofline_ms"] + rf_fwd["roofline_ms"],
            "flops_tflops": rf_bwd["flops_tflops"] + rf_fwd["flops_tflops"],
        }

    roofline_data = {"checkpointing_adjusted": uses_checkpointing}
    meas_fwd = get_measured_phase_ms(entry, "forward_ms")
    meas_bwd = get_measured_phase_ms(entry, "backward_ms")
    meas_opt = get_measured_phase_ms(entry, "optimizer_ms")

    for label, rf, meas_ms in [
        ("forward", rf_fwd, meas_fwd),
        ("backward", rf_bwd, meas_bwd),
        ("optimizer", rf_opt, meas_opt),
    ]:
        r_ms = rf["roofline_ms"]
        r_flops = rf["flops_tflops"]
        phase_data = {"roofline_ms": round(r_ms, 4), "flops_tflops": round(r_flops, 4)}
        if meas_ms and meas_ms > 0:
            phase_data["measured_ms"] = round(meas_ms, 3)
            phase_data["roofline_perc"] = round(r_ms / meas_ms * 100, 1)
            phase_data["mfu_perc"] = round(r_flops / (meas_ms / 1000) / peak_tflops * 100, 1)
        roofline_data[label] = phase_data

    rf_total_ms = rf_fwd["roofline_ms"] + rf_bwd["roofline_ms"] + rf_opt["roofline_ms"]
    rf_total_flops = rf_fwd["flops_tflops"] + rf_bwd["flops_tflops"] + rf_opt["flops_tflops"]
    meas_total = entry.get("throughput", {}).get("step_time_ms") or get_measured_phase_ms(entry, "total_ms")
    total_data = {
        "roofline_ms": round(rf_total_ms, 4),
        "flops_tflops": round(rf_total_flops, 4),
    }
    if meas_total and meas_total > 0:
        total_data["measured_ms"] = round(meas_total, 3)
        total_data["roofline_perc"] = round(rf_total_ms / meas_total * 100, 1)
        total_data["mfu_perc"] = round(rf_total_flops / (meas_total / 1000) / peak_tflops * 100, 1)
    roofline_data["total"] = total_data

    # --- CCL utilization ---
    ccl_data = {}
    eth_bw = hw.get("eth_bw_gb_s_per_link", 0)

    # DDP gradient sync
    theo_grad_sync_ms = ccl.get("grad_sync_time_ms", 0)
    if theo_grad_sync_ms > 0:
        meas_grad_sync_ms = get_measured_phase_ms(entry, "gradient_sync_ms")
        ccl_data["grad_sync_theoretical_ms"] = round(theo_grad_sync_ms, 4)
        if meas_grad_sync_ms and meas_grad_sync_ms > 0:
            util = theo_grad_sync_ms / meas_grad_sync_ms
            ccl_data["grad_sync_measured_ms"] = round(meas_grad_sync_ms, 3)
            ccl_data["grad_sync_bw_GBs"] = round(eth_bw * util, 2)
            ccl_data["grad_sync_util_perc"] = round(util * 100, 1)

    # TP CCL (forward / backward / total)
    for label, theo_key in [
        ("fwd", "ccl_forward_time_ms"),
        ("bwd", "ccl_backward_time_ms"),
        ("total", "ccl_total_tp_ms"),
    ]:
        theo_ms = ccl.get(theo_key, 0)
        if theo_ms <= 0:
            continue
        meas_ms = _get_measured_tp_ccl_ms(entry, label)
        ccl_data[f"{label}_ccl_theoretical_ms"] = round(theo_ms, 4)
        if meas_ms and meas_ms > 0:
            util = theo_ms / meas_ms
            ccl_data[f"{label}_ccl_measured_ms"] = round(meas_ms, 3)
            ccl_data[f"{label}_ccl_util_perc"] = round(util * 100, 1)

    if ccl_data:
        roofline_data["ccl"] = ccl_data

    entry["roofline"] = roofline_data


def main():
    parser = argparse.ArgumentParser(description="Add roofline analysis to experiment results")
    parser.add_argument("results_json", help="Path to extracted_results JSON")
    parser.add_argument("roofline_dir", help="Path to tt-train-roofline directory")
    parser.add_argument("-o", "--output", help="Output JSON (default: <input>_roofline.json)")
    parser.add_argument(
        "--hardware",
        default="bh_glx",
        help="Hardware config for roofline tool (default: bh_glx)",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help=(
            "Peak BF16 TFLOPs/s for MFU calculation. " "If not set, uses the value from the roofline hardware config."
        ),
    )
    args = parser.parse_args()

    results_path = Path(args.results_json)
    roofline_dir = Path(args.roofline_dir)
    output = (
        Path(args.output)
        if args.output
        else results_path.with_name(results_path.stem + "_roofline" + results_path.suffix)
    )

    if not roofline_dir.exists():
        print(f"Roofline directory not found: {roofline_dir}")
        sys.exit(1)

    results = json.loads(results_path.read_text())
    print(f"Loaded {len(results)} experiments\n")

    annotated = enrich_roofline(results, roofline_dir, args.hardware, args.peak_tflops)
    print(f"\n{annotated}/{len(results)} experiments annotated")

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    annotated = sum(1 for e in results if "roofline" in e)
    print(f"\n{annotated}/{len(results)} experiments annotated → {output}")


if __name__ == "__main__":
    main()
