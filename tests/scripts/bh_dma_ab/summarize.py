#!/usr/bin/env python3
# Temporary: summarize DMA on/off A/B results from run_dma_ab.sh.
import json
import os
import statistics
import sys


def load(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    for b in data.get("benchmarks", []):
        if b.get("aggregate_name") == "median" and "bytes_per_second" in b:
            out[b["run_name"].removesuffix("/real_time")] = b["bytes_per_second"] / 2**30
    return out


def table(title, on, off):
    print(f"### {title}\n")
    if not on or not off:
        print("missing results\n")
        return
    print("| benchmark | DMA on GiB/s | DMA off GiB/s | on/off |")
    print("|---|---:|---:|---:|")
    for name, v in on.items():
        if name in off:
            print(f"| {name} | {v:.3f} | {off[name]:.3f} | {v / off[name]:.2f}x |")
    print()


d = sys.argv[1]
table("Slow dispatch (write_core per page)", load(f"{d}/sd_dma_on.json"), load(f"{d}/sd_dma_off.json"))
table("Fast dispatch (control)", load(f"{d}/fd_dma_on.json"), load(f"{d}/fd_dma_off.json"))

init = os.path.join(d, "init_times.txt")
if os.path.exists(init):
    times = {"on": [], "off": []}
    with open(init) as f:
        for line in f:
            k, v = line.split()
            times[k].append(int(v))
    if times["on"] and times["off"]:
        print("### test_clean_init wall time (median of 5)\n")
        print(f"DMA on: {statistics.median(times['on'])} ms, DMA off: {statistics.median(times['off'])} ms\n")
