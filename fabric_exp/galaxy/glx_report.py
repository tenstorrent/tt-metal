#!/usr/bin/env python3
"""Galaxy comparison tables from one frs_glx.sbatch job (RESULTS §98).

    glx_report.py /data/aagarwal/frs-glx/jobs/<id>/results [--md]

For each suite (mesh, linear, ring) and test, side by side: main, mainka (main's router with the keepalive at 1000)
and v3q. Rate as payload GB/s per flow (from the summary CSV) and cycles/packet in the device's own clock (the slowest
sender's cycles over its packets, from the per-sender CSV). v3q's gain is over main.
"""
import argparse
import csv
import os
import re
import sys

CONFIGS = ["main", "mainka", "v3q"]
SUITES = ["mesh", "linear", "ring"]


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_name(name):
    """GlxMesh1HopEW_size_16_ntype_unicast_write -> (GlxMesh1HopEW, 16, unicast_write)."""
    m = re.match(r"(.+?)_size_(\d+)(?:_ntype_(\w+))?$", name)
    if m:
        return m.group(1), int(m.group(2)), m.group(3) or ""
    m = re.match(r"(.+?)_ntype_(\w+)_size_(\d+)$", name)
    if m:
        return m.group(1), int(m.group(3)), m.group(2)
    return name, None, ""


def rows_for(results, label):
    """{(test, size, ntype): [GB/s, cycles/packet]} for one run label, and the device clock (Hz).

    Cycles/packet is the clock over the summary's packets per second (the study's convention: 1.35e9 / pps on P150).
    The clock is recovered from the per-sender CSV, where pps = packets x traffic count x clock / cycles."""
    clocks = []
    for r in load(os.path.join(results, f"{label}_tests.csv")):
        n = int(r["num_packets"]) * int(r["total_traffic_count"])
        if n and int(r["cycles"]):
            clocks.append(float(r["packets_per_second"]) * int(r["cycles"]) / n)
    clock = sorted(clocks)[len(clocks) // 2] if clocks else None
    out = {}
    for r in load(os.path.join(results, f"{label}.csv")):
        test, size, ntype = parse_name(r["test_name"])
        size = size if size is not None else int(r["packet_size"])
        pps = float(r["avg_packets_per_s"])
        out[(test, size, ntype or r["ntype"].lower())] = [float(r["avg_bandwidth_gigabytes_per_s"]),
                                                         clock / pps if clock and pps else None]
    return out, clock


def status(results, label):
    log = os.path.join(results, f"{label}.log")
    if not os.path.exists(log):
        return "not run"
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(log, errors="replace").read())
    m = re.findall(r"Tests: (\d+) executed, (\d+) failed", text)
    if m:
        return f"{m[-1][0]} executed, {m[-1][1]} failed"
    for pat in ("TT_THROW", "TT_FATAL", "FATAL", "Timeout", "timed out", "hang"):
        if pat in text:
            return f"no summary ({pat} in log)"
    return "no summary"


def fmt(v, digits=1):
    return "—" if v is None else f"{v:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--md", action="store_true", help="markdown tables")
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated; the last is compared to the first")
    ap.add_argument("--suites", default=",".join(SUITES), help="comma-separated")
    a = ap.parse_args()
    configs, suites = a.configs.split(","), a.suites.split(",")

    print("Validation:", "; ".join(f"{l}: {status(a.results, l)}" for l in ("main_val", "v3q_val")))
    for suite in suites:
        got = {c: rows_for(a.results, f"{c}_{suite}") for c in configs}
        data = {c: got[c][0] for c in configs}
        clocks = ", ".join(f"{c} {got[c][1] / 1e6:.0f} MHz" for c in configs if got[c][1])
        print(f"\n## {suite}  (" + "; ".join(f"{c}: {status(a.results, f'{c}_{suite}')}" for c in configs) + ")")
        if clocks:
            print(f"   device clock: {clocks}")
        keys = []
        for c in configs:
            keys += [k for k in data[c] if k not in keys]
        if not keys:
            continue
        hdr = ["test", "size", "ntype"] + [f"{c} GB/s" for c in configs] + [f"{c} cyc/pkt" for c in configs] + [f"{configs[-1]}/{configs[0]}"]
        if a.md:
            print("| " + " | ".join(hdr) + " |")
            print("| " + " | ".join(["---"] * 3 + ["---:"] * (len(hdr) - 3)) + " |")
        else:
            print("  ".join(f"{h:>14}" if i > 2 else f"{h:<22}" if i == 0 else f"{h:>14}" for i, h in enumerate(hdr)))
        for k in keys:
            vals = [data[c].get(k) for c in configs]
            gb = [v[0] if v else None for v in vals]
            cp = [v[1] if v else None for v in vals]
            gain = f"{gb[-1] / gb[0]:.2f}x" if gb[0] and gb[-1] else "—"
            cells = [k[0], str(k[1]), k[2]] + [fmt(g, 2) for g in gb] + [fmt(c) for c in cp] + [gain]
            if a.md:
                print("| " + " | ".join(cells) + " |")
            else:
                print("  ".join(f"{x:<22}" if i == 0 else f"{x:>14}" for i, x in enumerate(cells)))


if __name__ == "__main__":
    sys.exit(main())
