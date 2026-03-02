#!/usr/bin/env python3
"""
Extract profiling results from experiment runs.

Parses profiler CSV files for each experiment, extracts per-device
per-iteration phase timings, and writes a consolidated JSON.

Usage:
    python extract_results.py <run_dir>                           # single run
    python extract_results.py run1/ run2/ -o merged.json          # merge multiple runs
    python extract_results.py run1/ --merge existing.json         # add to existing JSON
    python extract_results.py --csv /path/to/ops_perf.csv         # standalone CSV
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

MARKERS_PER_CALL = 10
DEFAULT_DEVICE_CLOCK_GHZ = 1.35
DEVICE_CLOCK_GHZ = DEFAULT_DEVICE_CLOCK_GHZ

PHASE_MAP = {
    ("dataloader_step_done", "forward_pass_done"): "forward_ms",
    ("forward_pass_done", "backward_pass_done"): "backward_ms",
    ("backward_pass_done", "gradient_sync_done"): "gradient_sync_ms",
    ("gradient_sync_done", "optimizer_step_done"): "optimizer_ms",
    ("optimizer_step_done", "iteration"): "other_ms",
    ("iteration", "dataloader_step_done"): "other_ms",
}

PHASE_COLS = [
    "forward_ms",
    "backward_ms",
    "gradient_sync_ms",
    "optimizer_ms",
    "other_ms",
]

CCL_OPS = {
    "ReduceScatterMinimalAsyncDeviceOperation": "rs",
    "AllGatherAsyncDeviceOperation": "ag",
}

PHASE_TO_CCL_PREFIX = {
    "forward_ms": "fwd",
    "backward_ms": "bwd",
    "gradient_sync_ms": "sync",
    "optimizer_ms": "opt",
}

CCL_COLS = [
    "fwd_rs_ms",
    "fwd_ag_ms",
    "bwd_rs_ms",
    "bwd_ag_ms",
    "sync_rs_ms",
    "sync_ag_ms",
    "opt_rs_ms",
    "opt_ag_ms",
]

DRAM_LINE_RE = re.compile(
    r"DRAM: Segment Peak ([\d.]+) MB, "
    r"Allocations ([\d.]+) MB, "
    r"Deallocations ([\d.]+) MB, "
    r"Segment Change ([+\-\d.]+) MB"
)
DRAM_CUM_RE = re.compile(
    r"DRAM: Cumulative Peak ([\d.]+) MB, " r"Cumulative Current ([\d.]+) MB"
)
FINAL_DRAM_RE = re.compile(
    r"Overall DRAM Peak: ([\d.]+) MB, Final DRAM Usage: ([\d.]+) MB"
)
PARAMS_RE = re.compile(r"Number of parameters:\s*(\d+)")
SEGMENT_RE = re.compile(r"^--- (\w+) ---$")
STEP_TIME_RE = re.compile(r"Full step time ([\d.]+) ms")
NAIVE_PROF_RE = re.compile(r"\[NAIVE_PROFILER\]\s+(\S+)\s+timestamp_us=(\d+)")


# ---------------------------------------------------------------------------
# stdout.log parsing
# ---------------------------------------------------------------------------


def parse_stdout(exp_dir: Path) -> dict:
    """Extract parameter count and memory segments from stdout.log."""
    stdout = exp_dir / "stdout.log"
    if not stdout.exists():
        return {}

    text = stdout.read_text()
    result = {}

    m = PARAMS_RE.search(text)
    if m:
        result["num_parameters"] = int(m.group(1))

    segments = {}
    current_segment = None
    for line in text.splitlines():
        seg_m = SEGMENT_RE.match(line.strip())
        if seg_m:
            current_segment = seg_m.group(1).lower()
            segments[current_segment] = {}
            continue

        if current_segment is None:
            continue

        dram_m = DRAM_LINE_RE.search(line)
        if dram_m:
            seg = segments[current_segment]
            seg["segment_peak_mb"] = float(dram_m.group(1))
            seg["allocations_mb"] = float(dram_m.group(2))
            seg["deallocations_mb"] = float(dram_m.group(3))
            seg["segment_change_mb"] = float(dram_m.group(4))

        cum_m = DRAM_CUM_RE.search(line)
        if cum_m:
            seg = segments[current_segment]
            seg["cumulative_peak_mb"] = float(cum_m.group(1))
            seg["cumulative_current_mb"] = float(cum_m.group(2))

    final_m = FINAL_DRAM_RE.search(text)
    if final_m:
        result["overall_dram_peak_mb"] = float(final_m.group(1))
        result["final_dram_usage_mb"] = float(final_m.group(2))

    if segments:
        result["memory_segments"] = segments

    # Step times (available with or without profiler)
    step_times = [float(m.group(1)) for m in STEP_TIME_RE.finditer(text)]
    if step_times:
        result["step_times_ms"] = step_times

    return result


def extract_step_timings(stdout_data: dict) -> dict | None:
    """Build a timings-like dict from stdout step times (for non-profiled runs).

    Skips iterations 1 (compilation) and 2 (warmup) for the average.
    """
    step_times = stdout_data.get("step_times_ms", [])
    if not step_times:
        return None

    iterations = []
    for i, t in enumerate(step_times):
        iterations.append({"iteration": i + 1, "total_step_ms": round(t, 3)})

    steady = iterations[2:] if len(iterations) > 2 else iterations[-1:]

    avg = {}
    if steady:
        avg["total_step_ms"] = round(
            sum(it["total_step_ms"] for it in steady) / len(steady), 3
        )

    return {
        "step_times": {
            "iterations": iterations,
            "average": avg,
            "num_steady_iterations": len(steady),
        }
    }


# ---------------------------------------------------------------------------
# Naive profiler parsing
# ---------------------------------------------------------------------------

NAIVE_PHASE_MAP = {
    ("dataloader_step_done", "forward_pass_done"): "forward_ms",
    ("forward_pass_done", "backward_pass_done"): "backward_ms",
    ("backward_pass_done", "gradient_sync_done"): "gradient_sync_ms",
    ("gradient_sync_done", "optimizer_step_done"): "optimizer_ms",
    ("optimizer_step_done", "iteration"): "other_ms",
    ("iteration", "dataloader_step_done"): "other_ms",
}


def extract_naive_timings(exp_dir: Path) -> dict | None:
    """Extract per-iteration phase timings from [NAIVE_PROFILER] markers in stdout."""
    stdout = exp_dir / "stdout.log"
    if not stdout.exists():
        return None

    markers = []
    for line in stdout.read_text().splitlines():
        m = NAIVE_PROF_RE.search(line)
        if m:
            ident = m.group(1)
            ts_us = int(m.group(2))
            markers.append((ident, ts_us))

    if len(markers) < 2:
        return None

    # Skip everything up to and including the first compilation step's markers.
    # The first "iteration_2" marks the start of steady-state.
    first_iter2 = None
    for i, (ident, _) in enumerate(markers):
        if ident == "iteration_2":
            first_iter2 = i
            break
    if first_iter2 is not None:
        markers = markers[first_iter2:]

    records = []
    for i in range(1, len(markers)):
        prev_ident, prev_ts = markers[i - 1]
        curr_ident, curr_ts = markers[i]

        prev_norm = re.sub(r"iteration_\d+", "iteration", prev_ident)
        curr_norm = re.sub(r"iteration_\d+", "iteration", curr_ident)

        pair = (prev_norm, curr_norm)
        if pair not in NAIVE_PHASE_MAP:
            continue

        duration_ms = (curr_ts - prev_ts) / 1000.0

        iter_num = None
        for s in [prev_ident, curr_ident]:
            m = re.match(r"iteration_(\d+)", s)
            if m:
                iter_num = int(m.group(1))
                break
        if iter_num is None:
            for j in range(i, len(markers)):
                m = re.match(r"iteration_(\d+)", markers[j][0])
                if m:
                    iter_num = int(m.group(1))
                    break
        if iter_num is None:
            continue

        records.append(
            {
                "iteration": iter_num,
                "phase": NAIVE_PHASE_MAP[pair],
                "duration_ms": duration_ms,
            }
        )

    if not records:
        return None

    df = pd.DataFrame(records)
    pivot = (
        df.pivot_table(
            index="iteration", columns="phase", values="duration_ms", aggfunc="sum"
        )
        .reindex(columns=PHASE_COLS, fill_value=0.0)
        .reset_index()
    )
    pivot["total_ms"] = pivot[PHASE_COLS].sum(axis=1)

    iterations = []
    for _, row in pivot.iterrows():
        entry = {"iteration": int(row["iteration"])}
        for col in PHASE_COLS + ["total_ms"]:
            entry[col] = round(float(row[col]), 3)
        iterations.append(entry)

    if len(iterations) > 1:
        steady = iterations[1:]
    else:
        steady = list(iterations)

    avg = {}
    if steady:
        for col in PHASE_COLS + ["total_ms"]:
            avg[col] = round(sum(it[col] for it in steady) / len(steady), 3)

    return {
        "device_host": {
            "iterations": iterations,
            "average": avg,
            "num_steady_iterations": len(steady),
        }
    }


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------


def find_csv(exp_dir: Path):
    """Find the profiler CSV: check copied report first, then parse stderr."""
    report_dir = exp_dir / "profiler_report"
    if report_dir.exists():
        csvs = list(report_dir.glob("ops_perf_results_*.csv"))
        if csvs:
            return csvs[0]

    stderr = exp_dir / "stderr.log"
    if stderr.exists():
        for line in stderr.read_text().splitlines():
            if "OPs csv generated at:" in line:
                path = Path(line.split("OPs csv generated at:")[-1].strip())
                if path.exists():
                    return path

    return None


# ---------------------------------------------------------------------------
# Timing extraction (adapted from profiler_results.ipynb)
# ---------------------------------------------------------------------------


def extract_timings(csv_path: Path) -> dict | None:
    """Parse profiler CSV and return per-device per-iteration phase timings."""
    raw = pd.read_csv(csv_path)

    # Drop everything up to (and including) compilation_finished marker
    comp_mask = (raw["OP CODE"] == "ProfilerNoopOperation") & raw[
        "ATTRIBUTES"
    ].str.contains("compilation_finished", na=False)
    comp_idx = raw.index[comp_mask]
    if comp_idx.empty:
        return None
    raw = raw.iloc[comp_idx[-1] + 1 :].reset_index(drop=True)

    # Tag markers
    raw["_is_marker"] = False
    raw["_ident"] = None
    noop = raw["OP CODE"] == "ProfilerNoopOperation"
    raw.loc[noop, "_ident"] = raw.loc[noop, "ATTRIBUTES"].str.extract(
        r"'identifier':\s*'([^']+)'", expand=False
    )
    raw.loc[raw["_ident"].notna(), "_is_marker"] = True

    # Collapse marker groups (MARKERS_PER_CALL consecutive rows with same id)
    mrows = raw[raw["_is_marker"]].copy().reset_index()
    groups = []
    i = 0
    while i + MARKERS_PER_CALL <= len(mrows):
        ident = mrows.iloc[i]["_ident"]
        grp = mrows.iloc[i : i + MARKERS_PER_CALL]
        if (grp["_ident"] == ident).all():
            groups.append(
                {
                    "ident": ident,
                    "first": int(grp["index"].iloc[0]),
                    "last": int(grp["index"].iloc[-1]),
                }
            )
            i += MARKERS_PER_CALL
        else:
            i += 1

    if not groups:
        return None

    device_ids = sorted(raw.loc[~raw["_is_marker"], "DEVICE ID"].dropna().unique())
    if not len(device_ids):
        device_ids = [None]

    # Pre-compute CCL op durations: time from this op's start to next op's start
    non_marker = raw[~raw["_is_marker"]].copy()
    non_marker["_ccl_type"] = non_marker["OP CODE"].map(CCL_OPS)
    non_marker["_next_start"] = non_marker.groupby("DEVICE ID")[
        "DEVICE FW START CYCLE"
    ].shift(-1)
    non_marker["_ccl_dur_ms"] = (
        non_marker["_next_start"] - non_marker["DEVICE FW START CYCLE"]
    ) / (DEVICE_CLOCK_GHZ * 1e6)
    raw = raw.join(non_marker[["_ccl_type", "_ccl_dur_ms"]])

    # Walk consecutive marker pairs and compute durations
    records = []
    for gi in range(1, len(groups)):
        prev_id = groups[gi - 1]["ident"]
        curr_id = groups[gi]["ident"]
        prev_norm = re.sub(r"iteration_\d+", "iteration", prev_id)
        curr_norm = re.sub(r"iteration_\d+", "iteration", curr_id)

        pair = (prev_norm, curr_norm)
        if pair not in PHASE_MAP:
            continue

        between = raw.iloc[groups[gi - 1]["last"] + 1 : groups[gi]["first"]]
        between = between[~between["_is_marker"]]

        # Determine iteration number
        iter_num = None
        for s in [prev_id, curr_id]:
            m = re.match(r"iteration_(\d+)", s)
            if m:
                iter_num = int(m.group(1))
                break
        if iter_num is None:
            for j in range(gi, len(groups)):
                m = re.match(r"iteration_(\d+)", groups[j]["ident"])
                if m:
                    iter_num = int(m.group(1))
                    break
        if iter_num is None:
            continue

        phase = PHASE_MAP[pair]
        ccl_prefix = PHASE_TO_CCL_PREFIX.get(phase)

        for dev_id in device_ids:
            ops = (
                between[between["DEVICE ID"] == dev_id]
                if dev_id is not None
                else between
            )
            if len(ops) >= 1:
                duration_ms = (
                    ops["DEVICE FW END CYCLE"].iloc[-1]
                    - ops["DEVICE FW START CYCLE"].iloc[0]
                ) / (DEVICE_CLOCK_GHZ * 1e6)
            else:
                duration_ms = 0.0

            rec = {
                "device_id": int(dev_id) if dev_id is not None else None,
                "iteration": iter_num,
                "phase": phase,
                "duration_ms": duration_ms,
            }

            # Sum CCL op durations within this phase
            if ccl_prefix:
                ccl_ops = ops[ops["_ccl_type"].notna()]
                for ccl_short in CCL_OPS.values():
                    col = f"{ccl_prefix}_{ccl_short}_ms"
                    rec[col] = float(
                        ccl_ops.loc[
                            ccl_ops["_ccl_type"] == ccl_short, "_ccl_dur_ms"
                        ].sum()
                    )

            records.append(rec)

    if not records:
        return None

    df = pd.DataFrame(records)

    all_cols = PHASE_COLS + ["total_ms"] + CCL_COLS
    result = {}
    for dev_id in device_ids:
        dev_key = f"device_{int(dev_id)}" if dev_id is not None else "device"
        dev_val = int(dev_id) if dev_id is not None else None
        dev_df = df[df["device_id"] == dev_val]

        pivot = (
            dev_df.pivot_table(
                index="iteration", columns="phase", values="duration_ms", aggfunc="sum"
            )
            .reindex(columns=PHASE_COLS, fill_value=0.0)
            .reset_index()
        )
        pivot["total_ms"] = pivot[PHASE_COLS].sum(axis=1)

        # Add CCL columns (summed directly, not pivoted by phase)
        for col in CCL_COLS:
            if col in dev_df.columns:
                ccl_by_iter = dev_df.groupby("iteration")[col].sum()
                pivot = pivot.merge(ccl_by_iter.rename(col), on="iteration", how="left")
                pivot[col] = pivot[col].fillna(0.0)
            else:
                pivot[col] = 0.0

        iterations = []
        for _, row in pivot.iterrows():
            entry = {"iteration": int(row["iteration"])}
            for col in all_cols:
                entry[col] = round(float(row.get(col, 0)), 3)
            iterations.append(entry)

        # Average over steady-state (exclude first iteration)
        if len(iterations) > 1:
            steady = [
                it for it in iterations if it["iteration"] > iterations[0]["iteration"]
            ]
        else:
            steady = list(iterations)

        avg = {}
        if steady:
            for col in all_cols:
                avg[col] = round(sum(it[col] for it in steady) / len(steady), 3)

        result[dev_key] = {
            "iterations": iterations,
            "average": avg,
            "num_steady_iterations": len(steady),
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_timings(timings: dict) -> None:
    """Pretty-print per-device phase timings to stdout."""
    for dev_key, dev_data in timings.items():
        avg = dev_data.get("average", {})
        iters = dev_data.get("iterations", [])

        print(f"\n  === {dev_key}: Per-Iteration Phase Timings (ms) ===")
        header = f"  {'iter':>4s}"
        for col in PHASE_COLS + ["total_ms"]:
            header += f"  {col:>16s}"
        print(header)

        for it in iters:
            row = f"  {it['iteration']:4d}"
            for col in PHASE_COLS + ["total_ms"]:
                row += f"  {it.get(col, 0):16.2f}"
            print(row)

        if avg:
            first_iter = iters[0]["iteration"] if iters else "?"
            print(f"\n  Average (excluding iteration {first_iter}):")
            for col in PHASE_COLS + ["total_ms"]:
                print(f"    {col:20s}: {avg.get(col, 0):10.2f} ms")


def main_csv(csv_path: Path) -> None:
    """Standalone mode: extract and print timings from a single CSV."""
    print(f"Parsing {csv_path} ...")
    timings = extract_timings(csv_path)
    if timings is None:
        print("No timing data found.")
        sys.exit(1)
    print_timings(timings)


def main_run_dir(
    run_dirs: list[Path],
    output: Path | None,
    merge_from: Path | None = None,
    seq_len: int = 2048,
) -> None:
    """Batch mode: process experiments from one or more run directories."""
    if output is None:
        output = run_dirs[0] / "extracted_results.json"

    exp_dirs = []
    for run_dir in run_dirs:
        found = sorted(
            d for d in run_dir.iterdir() if d.is_dir() and (d / "meta.json").exists()
        )
        print(f"Found {len(found)} experiments in {run_dir}")
        exp_dirs.extend(found)
    print()

    results = []
    for exp_dir in exp_dirs:
        meta = json.loads((exp_dir / "meta.json").read_text())
        name = meta["name"]
        profiler_mode = meta.get("profiler", True)
        stdout_data = parse_stdout(exp_dir)

        # 1. Tracy profiler CSV (highest fidelity)
        timings = None
        csv_path = None
        if profiler_mode is True:
            csv_path = find_csv(exp_dir)
            if csv_path is not None:
                print(f"  [{name}] {csv_path.name} ...", end=" ")
                try:
                    timings = extract_timings(csv_path)
                except Exception as e:
                    print(f"ERROR: {e}")
            else:
                print(
                    f"  [{name}] WARNING: profiled but no CSV, falling back ...",
                    end=" ",
                )

        # 2. Naive profiler (host-side timestamps, ~5% accuracy)
        naive_timings = None
        if timings is None:
            naive_timings = extract_naive_timings(exp_dir)

        # 3. Step times from stdout (always available)
        step_timings = extract_step_timings(stdout_data)

        # Throughput: tokens/sec and tokens/sec/device
        exp = meta.get("experiment", {})
        local_batch = exp.get("local_batch", 1)
        ddp = exp.get("ddp", 1)
        grad_accum = exp.get("grad_accum", 1)
        total_devices = meta.get("total_devices", 1)
        tokens_per_step = local_batch * seq_len * ddp * grad_accum

        # Best available step time for throughput.
        best_step_ms = None
        if timings:
            first_dev = next(iter(timings))
            best_step_ms = timings[first_dev].get("average", {}).get("total_ms")
        elif naive_timings:
            best_step_ms = (
                naive_timings["device_host"].get("average", {}).get("total_ms")
            )
        elif step_timings and profiler_mode is not True:
            best_step_ms = (
                step_timings["step_times"].get("average", {}).get("total_step_ms")
            )

        throughput = {}
        if best_step_ms and best_step_ms > 0:
            tps = tokens_per_step / (best_step_ms / 1000)
            throughput = {
                "tokens_per_step": tokens_per_step,
                "step_time_ms": round(best_step_ms, 3),
                "tokens_per_sec": round(tps, 1),
                "tokens_per_sec_per_device": round(tps / total_devices, 1),
            }

        entry = {
            "name": name,
            "command": meta.get("command"),
            "profiler": profiler_mode,
            "experiment": exp,
            "mesh_shape": meta.get("mesh_shape"),
            "total_devices": total_devices,
            "csv_path": str(csv_path) if csv_path else None,
            "num_parameters": stdout_data.get("num_parameters"),
            "memory": {
                "overall_dram_peak_mb": stdout_data.get("overall_dram_peak_mb"),
                "final_dram_usage_mb": stdout_data.get("final_dram_usage_mb"),
                "segments": stdout_data.get("memory_segments", {}),
            },
            "timings": timings,
            "naive_timings": naive_timings,
            "step_timings": step_timings,
            "throughput": throughput,
        }
        results.append(entry)

        if timings:
            first_dev = next(iter(timings))
            avg = timings[first_dev].get("average", {})
            print(
                f"fwd={avg.get('forward_ms', 0):.1f} "
                f"bwd={avg.get('backward_ms', 0):.1f} "
                f"sync={avg.get('gradient_sync_ms', 0):.1f} "
                f"opt={avg.get('optimizer_ms', 0):.1f} "
                f"total={avg.get('total_ms', 0):.1f} ms"
            )
        elif naive_timings:
            avg = naive_timings["device_host"].get("average", {})
            print(
                f"  [{name}] (naive profiler) "
                f"fwd={avg.get('forward_ms', 0):.1f} "
                f"bwd={avg.get('backward_ms', 0):.1f} "
                f"sync={avg.get('gradient_sync_ms', 0):.1f} "
                f"opt={avg.get('optimizer_ms', 0):.1f} "
                f"total={avg.get('total_ms', 0):.1f} ms"
            )
        elif step_timings:
            avg = step_timings["step_times"].get("average", {})
            print(
                f"  [{name}] step_time={avg.get('total_step_ms', 0):.1f} ms (no profiler)"
            )
        else:
            print(f"  [{name}] no timing data")

    # Dedup by name (last occurrence wins — later run_dirs override earlier ones)
    seen = {}
    for r in results:
        seen[r["name"]] = r
    if len(seen) < len(results):
        print(
            f"\n  Deduped: {len(results)} → {len(seen)} (later runs override earlier)"
        )
    results = list(seen.values())

    # Merge with existing results if requested
    if merge_from and merge_from.exists():
        existing = json.loads(merge_from.read_text())
        existing_names = {e["name"] for e in existing}
        new_names = {r["name"] for r in results}
        added = new_names - existing_names
        updated = new_names & existing_names

        merged = {e["name"]: e for e in existing}
        for r in results:
            merged[r["name"]] = r
        results = list(merged.values())

        if updated:
            print(f"  Updated {len(updated)} existing: {sorted(updated)}")
        if added:
            print(f"  Added {len(added)} new: {sorted(added)}")

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    ok = sum(
        1
        for r in results
        if r.get("timings") or r.get("naive_timings") or r.get("step_timings")
    )
    print(f"\n{ok}/{len(results)} experiments with data → {output}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract profiling results from experiment runs or a single CSV"
    )
    parser.add_argument(
        "run_dirs", nargs="*", help="Experiment run directories (batch mode)"
    )
    parser.add_argument(
        "--csv", type=str, help="Single profiler CSV file (standalone mode)"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON (default: <first_run_dir>/extracted_results.json)",
    )
    parser.add_argument(
        "--merge",
        type=str,
        help="Merge new results into this existing JSON (updates matching names, adds new ones)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length for throughput calculation (default: 2048)",
    )
    parser.add_argument(
        "--device-clock-ghz",
        type=float,
        default=DEFAULT_DEVICE_CLOCK_GHZ,
        help=f"Device clock speed in GHz for cycle→time conversion (default: {DEFAULT_DEVICE_CLOCK_GHZ})",
    )
    args = parser.parse_args()

    global DEVICE_CLOCK_GHZ
    DEVICE_CLOCK_GHZ = args.device_clock_ghz

    if args.csv:
        main_csv(Path(args.csv))
    elif args.run_dirs:
        output = Path(args.output) if args.output else None
        merge_from = Path(args.merge) if args.merge else None
        main_run_dir(
            [Path(d) for d in args.run_dirs], output, merge_from, seq_len=args.seq_len
        )
    else:
        parser.error("Provide run_dir(s) or --csv <path>")


if __name__ == "__main__":
    main()
