# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Summarise one capture of the dispatch_fabric2d perf worker for A/B comparison.

The worker's phases share op codes (both fabric2d transports are `DispatchFabric2dDeviceOperation`),
so the rows are split by the signposts the worker emits, not by op code. A launch is one ttnn call
across the mesh, one row per chip; the ops CSV carries GLOBAL CALL COUNT = launch base + device id
and the device log carries run host ID = launch base + PCIe slot (tools/tracy/process_ops_logs.py
sets global_call_count from run_host_id), so the base is the launch key in both. Every launch inside
a signposted phase is a measured one (the worker warms up before its first signpost), so none is
dropped: `n` is the number summarised.

Two figures per phase and per zone. "op us" is the slowest chip's DEVICE KERNEL DURATION per launch,
median over launches: what bounds the op. For a `dspf2d_*` zone, "mesh-slowest" is the same shape
(slowest core anywhere per launch, median over launches) and "per-chip" is the median over
(chip, launch) pairs of that chip's slowest stream core: the per-chip view, insensitive to one slow
chip, and not additive against the op time. Zone rows are limited to the launches the ops table
holds, so the two tables aggregate the same launches. The transport is read off the zones a launch
emits: only fan-out emits `dspf2d_mc_*`.

    python summarize_fabric2d_ab.py <ops_perf_results.csv | profiler subdir> [--label NAME] [--json OUT] [--op-code CODE]

A profiler subdir (`generated/profiler/<cell>`) holds `reports/<stamp>/ops_perf_results_<stamp>.csv`
and `.logs/profile_log_device.csv`; a bare CSV path is resolved against that same layout.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

KERNEL_NS = "DEVICE KERNEL DURATION [ns]"
FW_NS = "DEVICE FW DURATION [ns]"


def resolve_inputs(target: str):
    p = Path(target)
    if p.is_file():
        ops = p
        # <subdir>/reports/<stamp>/ops_perf_results_<stamp>.csv sits three levels under the subdir.
        logs = p.parent.parent.parent / ".logs" / "profile_log_device.csv"
    else:
        reports = sorted((p / "reports").iterdir())
        if not reports:
            sys.exit(f"no reports under {p}")
        stamp = reports[-1].name
        ops = reports[-1] / f"ops_perf_results_{stamp}.csv"
        logs = p / ".logs" / "profile_log_device.csv"
    if not logs.is_file():
        print(f"(no device log at {logs}; zones skipped)")
        return ops, None
    return ops, logs


def phases(df: pd.DataFrame):
    """(signpost name, rows until the next signpost), in capture order."""
    df = df.reset_index(drop=True)  # positional slicing below assumes a RangeIndex
    marks = df.index[df["OP TYPE"] == "signpost"].tolist()
    out = []
    for i, start in enumerate(marks):
        stop = marks[i + 1] if i + 1 < len(marks) else len(df)
        name = df.at[start, "OP CODE"]
        rows = df.iloc[start + 1 : stop]
        rows = rows[rows["OP TYPE"] != "signpost"]
        if len(rows):
            out.append((name, rows))
    return out


def per_launch_slowest_chip(rows: pd.DataFrame, col: str):
    """{launch key: max over chips}, launches in call order."""
    rows = rows[rows[col] != "-"].copy()
    rows[col] = rows[col].astype(float)
    launch = rows["GLOBAL CALL COUNT"].astype(int) - rows["DEVICE ID"].astype(int)
    return {int(k): g[col].max() for k, g in rows.groupby(launch, sort=True)}


def stats(values):
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize_ops(ops_csv: Path, op_code: str = ""):
    df = pd.read_csv(ops_csv)
    result = {}
    for name, rows in phases(df):
        if op_code:
            # A phase that interleaves several ops (the combine test tilizes and typecasts around
            # the op) is read one op code at a time, or the launch key mixes them.
            rows = rows[rows["OP CODE"] == op_code]
            if rows.empty:
                continue
        codes = sorted(rows["OP CODE"].unique())
        entry = {"op_codes": codes}
        by_launch = per_launch_slowest_chip(rows, KERNEL_NS)
        if not by_launch:
            entry["incomplete"] = "no device durations in this phase"
            result[name] = entry
            continue
        entry["launches"] = sorted(by_launch)
        entry["kernel_us"] = stats([v / 1e3 for v in by_launch.values()])
        fw = per_launch_slowest_chip(rows, FW_NS)
        entry["fw_us"] = stats([v / 1e3 for v in fw.values()]) if fw else None
        result[name] = entry
    return result


def summarize_zones(device_csv: Path, launches: set):
    """Zones for the launches the ops table holds, so both tables aggregate the same population."""
    with open(device_csv) as f:
        header = f.readline()
    freq_mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    df = pd.read_csv(device_csv, skiprows=1)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["zone name"].astype(str).str.startswith("dspf2d_")]
    if df.empty:
        return freq_mhz, {}
    key = ["PCIe slot", "core_x", "core_y", "RISC processor type", "run host ID", "zone name"]
    starts = df[df["type"] == "ZONE_START"].groupby(key)["time[cycles since reset]"].min()
    ends = df[df["type"] == "ZONE_END"].groupby(key)["time[cycles since reset]"].max()
    dur = (ends - starts).dropna() / freq_mhz  # cycles / MHz = us
    dur = dur.reset_index().rename(columns={"time[cycles since reset]": "us"})
    # run host ID is launch base + device id, like the ops CSV's GLOBAL CALL COUNT.
    dur["launch"] = dur["run host ID"].astype(int) - dur["PCIe slot"].astype(int)
    dur = dur[dur["launch"].isin(launches)]
    if dur.empty:
        return freq_mhz, {}
    # The transport a launch ran is written in its zone names: only fan-out emits dspf2d_mc_*.
    mc_launches = set(dur[dur["zone name"].str.startswith("dspf2d_mc_")]["launch"])
    dur["transport"] = dur["launch"].map(lambda r: "multicast" if r in mc_launches else "unicast")
    per_chip = dur.groupby(["transport", "zone name", "launch", "PCIe slot"])["us"].max().reset_index()
    per_launch = per_chip.groupby(["transport", "zone name", "launch"])["us"].max().reset_index()
    out = {}
    for (transport, zone), g in per_launch.groupby(["transport", "zone name"]):
        chips = per_chip[(per_chip["transport"] == transport) & (per_chip["zone name"] == zone)]
        entry = stats(g.sort_values("launch")["us"].tolist())
        entry["chip_launch_median"] = float(chips["us"].median())
        out.setdefault(transport, {})[zone] = entry
    return freq_mhz, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--op-code", default="", help="read only this op code out of each phase")
    args = ap.parse_args()
    ops_csv, device_csv = resolve_inputs(args.target)
    summary = {"label": args.label, "ops_csv": str(ops_csv), "phases": summarize_ops(ops_csv, args.op_code)}

    print(f"## {args.label or ops_csv}")
    print()
    print("op us = slowest chip per launch, median over launches (min/max over launches); fw us likewise.")
    print()
    print("| phase | op code | launches | op us | min | max | fw us |")
    print("|---|---|--:|--:|--:|--:|--:|")
    all_launches = set()
    for name, e in summary["phases"].items():
        if "incomplete" in e:
            print(f"| {name} | {', '.join(e['op_codes'])} | 0 | - | - | - | - |  ({e['incomplete']})")
            continue
        all_launches.update(e["launches"])
        k, fw = e["kernel_us"], e["fw_us"]
        fw_s = f"{fw['median']:.1f}" if fw else "-"
        print(
            f"| {name} | {', '.join(e['op_codes'])} | {k['n']} | {k['median']:.1f} | {k['min']:.1f} | {k['max']:.1f} | {fw_s} |"
        )

    if device_csv is not None:
        freq, zones = summarize_zones(device_csv, all_launches)
        summary["chip_freq_mhz"] = freq
        summary["zones"] = zones
        if zones:
            print()
            print("mesh-slowest = slowest core anywhere per launch, median over launches (min/max over launches);")
            print("per-chip = median over (chip, launch) of that chip's slowest stream core. Same launches as above.")
            print()
            print("| transport | zone | launches | mesh-slowest us | min | max | per-chip us |")
            print("|---|---|--:|--:|--:|--:|--:|")
            for transport in sorted(zones):
                for zone, z in zones[transport].items():
                    print(
                        f"| {transport} | {zone} | {z['n']} | {z['median']:.1f} | {z['min']:.1f} | {z['max']:.1f} "
                        f"| {z['chip_launch_median']:.1f} |"
                    )
        else:
            print()
            print("(device log holds no dspf2d_* zones for these launches)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
