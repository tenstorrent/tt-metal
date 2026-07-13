# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Measure DRAM read bandwidth of ttnn.bh_dram_read via the on-device profiler.
# Run with:  TT_METAL_DEVICE_PROFILER=1 python3 measure_bh_dram_read_bw.py
#
# Method: run the op on a large DRAM-interleaved tensor, close the device (which
# flushes the device profiler CSV), then read the "DRAM_READ" kernel zone cycle
# counts. Aggregate bytes/cycle = total_bytes / (slowest core's cycles), since
# the per-bank reader cores run concurrently.

import os
import sys
import re
import csv
from pathlib import Path

import torch
import ttnn

BH_DRAM_BW_GB_S = 512.0  # tt-metal microbenchmark spec constant for Blackhole

HEIGHT = 8192
WIDTH = 8192
ITERS = 3


def find_csv():
    base = os.environ.get("TT_METAL_PROFILER_DIR")
    home = os.environ.get("TT_METAL_HOME", os.getcwd())
    candidates = []
    if base:
        candidates.append(Path(base) / ".logs" / "profile_log_device.csv")
    candidates.append(Path(home) / "generated" / "profiler" / ".logs" / "profile_log_device.csv")
    for c in candidates:
        if c.exists():
            return c
    return None


def run_op():
    torch.manual_seed(0)
    t = torch.rand((HEIGHT, WIDTH), dtype=torch.bfloat16)
    dev = ttnn.open_device(device_id=0)
    try:
        x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for _ in range(ITERS):
            ttnn.bh_dram_read(x)
        ttnn.synchronize_device(dev)
    finally:
        ttnn.close_device(dev)  # flushes the device profiler CSV


def parse_csv(path):
    # First line is a comment containing CHIP_FREQ[MHz]; second line is the header.
    text = path.read_text().splitlines()
    freq_mhz = None
    for line in text[:3]:
        m = re.search(r"CHIP_FREQ\[MHz\]:\s*([0-9.]+)", line)
        if m:
            freq_mhz = float(m.group(1))
    # locate header row
    header_idx = next(i for i, l in enumerate(text) if "zone name" in l.lower())
    reader = csv.reader(text[header_idx:])
    rows = list(reader)
    header = [h.strip() for h in rows[0]]

    def col(name):
        for i, h in enumerate(header):
            if h.lower() == name.lower():
                return i
        return None

    ci = {k: col(k) for k in ["core_x", "core_y", "time[cycles since reset]", "zone name", "type"]}
    # per-core list of durations of the DRAM_READ zone
    open_ts = {}
    durations = {}
    for r in rows[1:]:
        if len(r) <= max(v for v in ci.values() if v is not None):
            continue
        zone = r[ci["zone name"]].strip()
        if "DRAM_READ" not in zone:
            continue
        core = (r[ci["core_x"]].strip(), r[ci["core_y"]].strip())
        cyc = int(r[ci["time[cycles since reset]"]].strip())
        typ = r[ci["type"]].strip().upper()
        if "BEGIN" in typ or "START" in typ or typ == "ZONE_START":
            open_ts[core] = cyc
        elif ("END" in typ or typ == "ZONE_END") and core in open_ts:
            durations.setdefault(core, []).append(cyc - open_ts.pop(core))
    return freq_mhz, durations


def main():
    if not os.environ.get("TT_METAL_DEVICE_PROFILER"):
        print("ERROR: set TT_METAL_DEVICE_PROFILER=1", file=sys.stderr)
        sys.exit(2)

    run_op()
    csv_path = find_csv()
    if not csv_path:
        print("ERROR: profiler CSV not found", file=sys.stderr)
        sys.exit(1)

    freq_mhz, durations = parse_csv(csv_path)
    if not durations:
        print("ERROR: no DRAM_READ zones found in CSV", file=sys.stderr)
        sys.exit(1)

    # bytes
    tile_bytes = 32 * 32 * 2  # bf16 tile
    num_tiles = (HEIGHT // 32) * (WIDTH // 32)
    total_bytes = num_tiles * tile_bytes

    # per-core best (min) cycles, then the binding core is the slowest of those
    per_core_best = {c: min(ds) for c, ds in durations.items()}
    num_cores = len(per_core_best)
    slowest_cycles = max(per_core_best.values())
    per_core_bytes = total_bytes / num_cores

    agg_bpc = total_bytes / slowest_cycles
    per_core_bpc = per_core_bytes / slowest_cycles

    print("\n================ bh_dram_read DRAM read bandwidth ================")
    print(f"tensor                : {HEIGHT}x{WIDTH} bf16")
    print(f"total bytes           : {total_bytes:,} ({total_bytes/1024/1024:.1f} MiB)")
    print(f"num tiles             : {num_tiles:,}   (page = {tile_bytes} B)")
    print(f"reader cores (banks)  : {num_cores}")
    print(f"AICLK                 : {freq_mhz:.1f} MHz" if freq_mhz else "AICLK: unknown")
    print(f"slowest core cycles   : {slowest_cycles:,}")
    print("-----------------------------------------------------------------")
    print(f"per-core bytes/cycle  : {per_core_bpc:.2f} B/cyc")
    print(f"AGGREGATE bytes/cycle : {agg_bpc:.2f} B/cyc")
    if freq_mhz:
        gbs = agg_bpc * freq_mhz * 1e6 / 1e9
        peak_bpc = BH_DRAM_BW_GB_S * 1e9 / (freq_mhz * 1e6)
        print(f"AGGREGATE bandwidth   : {gbs:.1f} GB/s")
        print(f"theoretical peak      : {BH_DRAM_BW_GB_S:.0f} GB/s  ({peak_bpc:.1f} B/cyc aggregate)")
        print(f"UTILIZATION           : {100.0*gbs/BH_DRAM_BW_GB_S:.1f} %")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
