#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-iteration perf counter analysis for fused ops.

Parses ITERATION zones emitted by kernel_codegen.py and computes FPU/SFPU/MATH
utilization metrics for each iteration on each device.

Usage:
    # Run a test with profiling enabled:
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILE_PERF_COUNTERS=47 \
        TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
        pytest tests/blaze/fused_ops/llama31_decoder_layer/test_llama31_decoder_layer.py -k "device_params0"

    # Analyze the results:
    python -m tracy.per_iteration_analysis -o per_iteration_metrics.csv

The script parses profile_log_device.csv and:
    1. Finds all ITERATION zones (zone_name="ITERATION", type=ZONE_START/ZONE_END)
    2. Groups perf counter events (id=9090) by their containing ITERATION zone
    3. Computes FPU/SFPU/MATH utilization = counter_value / ref_cnt * 100
    4. Outputs per-device, per-iteration metrics to CSV
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

try:
    from tracy.perf_counter_analysis import COUNTER_TYPE_NAMES
    from tracy.common import PROFILER_LOGS_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from perf_counter_analysis import COUNTER_TYPE_NAMES
    from common import PROFILER_LOGS_DIR

PERF_COUNTER_ID = 9090


@dataclass
class IterationMetrics:
    """Aggregated metrics for one iteration on one device."""

    device_id: int
    iteration: int
    fpu_util_min: float = 0.0
    fpu_util_max: float = 0.0
    fpu_util_avg: float = 0.0
    sfpu_util_min: float = 0.0
    sfpu_util_max: float = 0.0
    sfpu_util_avg: float = 0.0
    math_util_min: float = 0.0
    math_util_max: float = 0.0
    math_util_avg: float = 0.0
    num_cores: int = 0
    duration_cycles: int = 0
    # NoC/DRAM metrics come from simulating this iteration's NoC trace slice through tt-npe
    # (there is no hardware counter for them). None => not measured, which is distinct from
    # zero: it means either tt-npe was unavailable or the slice had no NoC events at all.
    noc_util: Optional[float] = None
    mcast_noc_util: Optional[float] = None
    dram_bw_util: Optional[float] = None
    congestion_impact: Optional[float] = None


def parse_device_log(log_path: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Parse profile_log_device.csv and return (device_info, dataframe)."""
    with open(log_path, "r") as f:
        first_line = f.readline()

    if "ARCH" in first_line:
        info = first_line.split(",")
        arch = info[0].split(":")[-1].strip()
        freq = int(info[1].split(":")[-1].strip())
        max_cores = int(info[2].split(":")[-1].strip()) if len(info) > 2 else None
    else:
        arch, freq, max_cores = "unknown", 1200, None

    device_info = {"arch": arch, "freq": freq, "max_compute_cores": max_cores}
    df = pd.read_csv(log_path, skiprows=1, header=0, na_filter=False)
    return device_info, df


def extract_iterations_and_counters(df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """Extract ITERATION zones and perf counter events from the dataframe.

    Returns:
        (iterations, perf_events) where each iteration has start/end timestamps
        and perf_events are the raw counter readings.
    """
    # CSV column indices (0-indexed after header)
    COL_CHIP_ID = 0
    COL_CORE_X = 1
    COL_CORE_Y = 2
    COL_RISC = 3
    COL_TIMER_ID = 4
    COL_TIMESTAMP = 5
    COL_ZONE_NAME = 10
    COL_ZONE_TYPE = 11
    COL_META_DATA = 14

    iterations = []
    perf_events = []

    # Track open ITERATION zones per (chip, core_x, core_y)
    open_iterations: Dict[Tuple[int, int, int], Dict] = {}
    iteration_counters: Dict[Tuple[int, int, int], int] = {}

    for row in df.itertuples(index=False):
        chip_id = row[COL_CHIP_ID]
        core_x = row[COL_CORE_X]
        core_y = row[COL_CORE_Y]
        risc = str(row[COL_RISC])
        timer_id = row[COL_TIMER_ID]
        timestamp = row[COL_TIMESTAMP]
        zone_name = str(row[COL_ZONE_NAME])
        zone_type = str(row[COL_ZONE_TYPE])
        meta_data = row[COL_META_DATA]

        core_key = (chip_id, core_x, core_y)

        # Handle ITERATION zones (only BRISC emits them)
        if zone_name == "ITERATION" and "BRISC" in risc:
            if zone_type == "ZONE_START":
                if core_key not in iteration_counters:
                    iteration_counters[core_key] = 0
                open_iterations[core_key] = {
                    "chip_id": chip_id,
                    "core_x": core_x,
                    "core_y": core_y,
                    "iteration": iteration_counters[core_key],
                    "start_ts": timestamp,
                    "end_ts": None,
                    "counters": {},
                }
            elif zone_type == "ZONE_END" and core_key in open_iterations:
                iter_data = open_iterations[core_key]
                iter_data["end_ts"] = timestamp
                iterations.append(iter_data)
                iteration_counters[core_key] += 1
                del open_iterations[core_key]

        # Handle perf counter events
        if timer_id == PERF_COUNTER_ID and meta_data:
            try:
                meta_str = str(meta_data).replace(";", ",").replace("'", '"')
                meta_dict = json.loads(meta_str)
            except (json.JSONDecodeError, AttributeError):
                continue

            counter_type_raw = meta_dict.get("counter type", 0)
            if isinstance(counter_type_raw, str):
                counter_type = counter_type_raw
            else:
                counter_type = COUNTER_TYPE_NAMES.get(counter_type_raw, f"UNKNOWN_{counter_type_raw}")

            perf_events.append(
                {
                    "chip_id": chip_id,
                    "core_x": core_x,
                    "core_y": core_y,
                    "timestamp": timestamp,
                    "counter_type": counter_type,
                    "value": meta_dict.get("value", 0),
                    "ref_cnt": meta_dict.get("ref cnt", 0),
                }
            )

    return iterations, perf_events


def assign_counters_to_iterations(iterations: List[Dict], perf_events: List[Dict]) -> None:
    """Assign each perf counter event to its containing ITERATION zone."""
    # Index iterations by core for efficient lookup
    by_core: Dict[Tuple[int, int, int], List[Dict]] = defaultdict(list)
    for it in iterations:
        key = (it["chip_id"], it["core_x"], it["core_y"])
        by_core[key].append(it)

    # Sort each core's iterations by start_ts
    for key in by_core:
        by_core[key].sort(key=lambda x: x["start_ts"])

    # Assign events to iterations
    for ev in perf_events:
        key = (ev["chip_id"], ev["core_x"], ev["core_y"])
        if key not in by_core:
            continue

        ts = ev["timestamp"]
        for it in by_core[key]:
            if it["start_ts"] <= ts and (it["end_ts"] is None or ts <= it["end_ts"]):
                ctype = ev["counter_type"]
                if ctype not in it["counters"]:
                    it["counters"][ctype] = {"value": 0, "ref_cnt": 0}
                it["counters"][ctype]["value"] += ev["value"]
                it["counters"][ctype]["ref_cnt"] = max(it["counters"][ctype]["ref_cnt"], ev["ref_cnt"])
                break


def compute_utilization(counters: Dict[str, Dict]) -> Dict[str, float]:
    """Compute utilization percentages from counter values."""
    result = {}
    for ctype in ["FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"]:
        if ctype in counters:
            val = counters[ctype]["value"]
            ref = counters[ctype]["ref_cnt"]
            if ref > 0:
                result[ctype] = val / ref * 100
            else:
                result[ctype] = 0.0
    return result


def aggregate_iterations(iterations: List[Dict]) -> List[IterationMetrics]:
    """Aggregate per-core iterations into per-device, per-iteration summaries."""
    # Group by (chip_id, iteration)
    grouped: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for it in iterations:
        key = (it["chip_id"], it["iteration"])
        grouped[key].append(it)

    summaries = []
    for (chip_id, iteration), core_iterations in sorted(grouped.items()):
        fpu_utils = []
        sfpu_utils = []
        math_utils = []
        durations = []

        for it in core_iterations:
            utils = compute_utilization(it["counters"])
            if "FPU_COUNTER" in utils:
                fpu_utils.append(utils["FPU_COUNTER"])
            if "SFPU_COUNTER" in utils:
                sfpu_utils.append(utils["SFPU_COUNTER"])
            if "MATH_COUNTER" in utils:
                math_utils.append(utils["MATH_COUNTER"])
            if it["end_ts"] is not None:
                durations.append(it["end_ts"] - it["start_ts"])

        summary = IterationMetrics(
            device_id=chip_id,
            iteration=iteration,
            fpu_util_min=min(fpu_utils) if fpu_utils else 0.0,
            fpu_util_max=max(fpu_utils) if fpu_utils else 0.0,
            fpu_util_avg=sum(fpu_utils) / len(fpu_utils) if fpu_utils else 0.0,
            sfpu_util_min=min(sfpu_utils) if sfpu_utils else 0.0,
            sfpu_util_max=max(sfpu_utils) if sfpu_utils else 0.0,
            sfpu_util_avg=sum(sfpu_utils) / len(sfpu_utils) if sfpu_utils else 0.0,
            math_util_min=min(math_utils) if math_utils else 0.0,
            math_util_max=max(math_utils) if math_utils else 0.0,
            math_util_avg=sum(math_utils) / len(math_utils) if math_utils else 0.0,
            num_cores=len(core_iterations),
            duration_cycles=int(sum(durations) / len(durations)) if durations else 0,
        )
        summaries.append(summary)

    return summaries


def analyze_per_iteration_metrics(
    log_dir: Path,
    noc_trace_dir: Optional[Path] = None,
) -> List[IterationMetrics]:
    """Main entry point: analyze profiler logs and return per-iteration metrics.

    If ``noc_trace_dir`` is given (and tt-npe is importable), NoC traces there are sliced on
    ITERATION boundaries and simulated, adding NoC/DRAM columns to each row.
    """
    log_dir = Path(log_dir)

    # Find profile_log_device.csv
    candidates = [
        log_dir / "profile_log_device.csv",
        log_dir / ".logs" / "profile_log_device.csv",
        log_dir.parent / "profile_log_device.csv",
    ]
    device_log = None
    for p in candidates:
        if p.exists():
            device_log = p
            break

    if device_log is None:
        logger.warning(f"profile_log_device.csv not found in {log_dir}")
        return []

    logger.info(f"Parsing {device_log}")
    device_info, df = parse_device_log(device_log)
    logger.info(f"Device: {device_info['arch']} @ {device_info['freq']}MHz, {len(df)} rows")

    iterations, perf_events = extract_iterations_and_counters(df)
    logger.info(f"Found {len(iterations)} ITERATION zones, {len(perf_events)} perf counter events")

    if not iterations:
        logger.warning("No ITERATION zones found - ensure kernel_codegen.py emits them")
        return []

    if not perf_events:
        logger.warning("No perf counter events - ensure TT_METAL_PROFILE_PERF_COUNTERS is set")
        return []

    assign_counters_to_iterations(iterations, perf_events)
    summaries = aggregate_iterations(iterations)
    logger.info(f"Aggregated to {len(summaries)} (device, iteration) summaries")

    if noc_trace_dir is not None:
        noc_metrics = collect_noc_metrics(Path(noc_trace_dir))
        if noc_metrics:
            matched = attach_noc_metrics(summaries, noc_metrics)
            logger.info(f"Attached NoC/DRAM metrics to {matched}/{len(summaries)} rows")

    return summaries


def collect_noc_metrics(noc_trace_dir: Path) -> Dict[Tuple[int, int], Dict[str, Optional[float]]]:
    """Merge NoC traces across devices, slice by ITERATION, and simulate each slice.

    Reuses tt-npe's own fabric merge so multi-device traces are combined exactly as the
    standard per-op flow does; zone markers survive that merge, which is what makes slicing
    the merged file possible.
    """
    try:
        from noc_per_iteration import ITERATION_ZONE, analyze_noc_per_iteration, import_npe
    except ImportError:
        from tracy.noc_per_iteration import ITERATION_ZONE, analyze_noc_per_iteration, import_npe

    npe, TopologyGraph = import_npe()
    if npe is None:
        return {}

    topology = noc_trace_dir / "topology.json"
    if not topology.exists():
        logger.warning(f"No topology.json in {noc_trace_dir}; cannot run tt-npe")
        return {}

    merged = sorted(noc_trace_dir.glob("noc_trace*_merged.json"))
    if not merged:
        merged = _merge_noc_traces(noc_trace_dir, topology, TopologyGraph)
    if not merged:
        logger.warning(f"No NoC traces to analyze in {noc_trace_dir}")
        return {}

    combined: Dict[Tuple[int, int], Dict[str, Optional[float]]] = {}
    for merged_trace in merged:
        combined.update(
            analyze_noc_per_iteration(
                merged_trace=merged_trace,
                topology_json=topology,
                output_dir=noc_trace_dir,
                zone_name=ITERATION_ZONE,
            )
        )
    return combined


def _merge_noc_traces(noc_trace_dir: Path, topology: Path, TopologyGraph) -> List[Path]:
    """Fabric-merge the per-device traces using tt-npe's own merger."""
    try:
        from fabric_post_process import process_traces
    except ImportError:
        logger.warning("tt-npe's fabric_post_process unavailable; cannot merge NoC traces")
        return []

    per_device = [p for p in sorted(noc_trace_dir.glob("noc_trace*.json")) if "_merged" not in p.name]
    if not per_device:
        return []

    out = noc_trace_dir / "noc_trace_iterations_merged.json"
    try:
        process_traces(TopologyGraph(str(topology)), [str(p) for p in per_device], str(out), True, True)
    except Exception as e:
        logger.error(f"Fabric merge failed: {e!r}")
        return []
    return [out] if out.exists() else []


def _fmt_optional(value: Optional[float]) -> str:
    """Render an optional metric: blank when not measured, never 'nan'."""
    return "" if value is None else f"{value:.2f}"


def _pct(value: Optional[float]) -> str:
    """Render an optional metric for the console table; '-' when not measured."""
    return "-" if value is None else f"{value:.2f}%"


def attach_noc_metrics(
    summaries: List[IterationMetrics],
    noc_metrics: Dict[Tuple[int, int], Dict[str, Optional[float]]],
) -> int:
    """Join tt-npe per-iteration results onto the perf-counter rows by (device, iteration)."""
    matched = 0
    for s in summaries:
        metrics = noc_metrics.get((s.device_id, s.iteration))
        if not metrics:
            continue
        s.noc_util = metrics.get("noc_util")
        s.mcast_noc_util = metrics.get("mcast_noc_util")
        s.dram_bw_util = metrics.get("dram_bw_util")
        s.congestion_impact = metrics.get("congestion_impact")
        matched += 1
    return matched


def write_csv(summaries: List[IterationMetrics], output_path: Path) -> None:
    """Write per-iteration metrics to CSV."""
    fieldnames = [
        "device_id",
        "iteration",
        "fpu_util_min",
        "fpu_util_max",
        "fpu_util_avg",
        "sfpu_util_min",
        "sfpu_util_max",
        "sfpu_util_avg",
        "math_util_min",
        "math_util_max",
        "math_util_avg",
        "num_cores",
        "duration_cycles",
        "noc_util",
        "mcast_noc_util",
        "dram_bw_util",
        "congestion_impact",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "device_id": s.device_id,
                    "iteration": s.iteration,
                    "fpu_util_min": f"{s.fpu_util_min:.2f}",
                    "fpu_util_max": f"{s.fpu_util_max:.2f}",
                    "fpu_util_avg": f"{s.fpu_util_avg:.2f}",
                    "sfpu_util_min": f"{s.sfpu_util_min:.2f}",
                    "sfpu_util_max": f"{s.sfpu_util_max:.2f}",
                    "sfpu_util_avg": f"{s.sfpu_util_avg:.2f}",
                    "math_util_min": f"{s.math_util_min:.2f}",
                    "math_util_max": f"{s.math_util_max:.2f}",
                    "math_util_avg": f"{s.math_util_avg:.2f}",
                    "num_cores": s.num_cores,
                    "duration_cycles": s.duration_cycles,
                    # Blank, not "nan"/"0": an unmeasured iteration must not read as an idle one.
                    "noc_util": _fmt_optional(s.noc_util),
                    "mcast_noc_util": _fmt_optional(s.mcast_noc_util),
                    "dram_bw_util": _fmt_optional(s.dram_bw_util),
                    "congestion_impact": _fmt_optional(s.congestion_impact),
                }
            )
    logger.info(f"Wrote {len(summaries)} rows to {output_path}")


def print_summary(summaries: List[IterationMetrics]) -> None:
    """Print a summary table."""
    if not summaries:
        print("No per-iteration metrics available")
        return

    print("\n" + "=" * 100)
    print("PER-ITERATION METRICS SUMMARY")
    print("=" * 100)

    by_device: Dict[int, List[IterationMetrics]] = defaultdict(list)
    for s in summaries:
        by_device[s.device_id].append(s)

    for device_id, device_summaries in sorted(by_device.items()):
        print(f"\nDevice {device_id} ({len(device_summaries)} iterations):")
        print("-" * 116)
        print(
            f"{'Iter':<6} {'FPU Avg':>10} {'SFPU Avg':>10} {'MATH Avg':>10} "
            f"{'NoC Util':>10} {'DRAM BW':>10} {'Cong':>8} {'Cores':>8} {'Cycles':>15}"
        )
        print("-" * 116)

        for s in sorted(device_summaries, key=lambda x: x.iteration):
            print(
                f"{s.iteration:<6} "
                f"{s.fpu_util_avg:>9.2f}% "
                f"{s.sfpu_util_avg:>9.2f}% "
                f"{s.math_util_avg:>9.2f}% "
                f"{_pct(s.noc_util):>10} "
                f"{_pct(s.dram_bw_util):>10} "
                f"{_pct(s.congestion_impact):>8} "
                f"{s.num_cores:>8} "
                f"{s.duration_cycles:>15,}"
            )

        if len(device_summaries) > 1:
            avg_fpu = sum(s.fpu_util_avg for s in device_summaries) / len(device_summaries)
            avg_sfpu = sum(s.sfpu_util_avg for s in device_summaries) / len(device_summaries)
            avg_math = sum(s.math_util_avg for s in device_summaries) / len(device_summaries)
            avg_cycles = sum(s.duration_cycles for s in device_summaries) / len(device_summaries)
            print("-" * 90)
            print(
                f"{'AVG':<6} "
                f"{avg_fpu:>9.2f}% "
                f"{avg_sfpu:>9.2f}% "
                f"{avg_math:>9.2f}% "
                f"{'':<8} "
                f"{int(avg_cycles):>15,}"
            )

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-iteration FPU/SFPU/MATH utilization from Tracy profiler logs"
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        nargs="?",
        default=Path("generated/profiler/.logs"),
        help="Directory containing profile_log_device.csv",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("per_iteration_metrics.csv"))
    parser.add_argument(
        "--noc-trace-dir",
        type=Path,
        default=None,
        help=(
            "Directory of NoC traces (plus topology.json) to slice per iteration and simulate "
            "with tt-npe, adding NoC/DRAM columns. Requires tt-npe on PYTHONPATH "
            "(branch snadeem/blaze_layer_support_v2 - main rejects FABRIC_2D_TORUS_X)."
        ),
    )
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    summaries = analyze_per_iteration_metrics(args.log_dir, noc_trace_dir=args.noc_trace_dir)

    if not summaries:
        return 1

    if not args.no_summary:
        print_summary(summaries)

    write_csv(summaries, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
