# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Align the newest profiler CSV to gate_up_sweep_manifest.json and print the table.

The sweep test runs each arm REPS times in a known order; matmul rows land in the
CSV in that same order. CORE COUNT is cross-checked per arm so a misalignment
raises instead of silently reporting the wrong config's time.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
import sys

MANIFEST = "generated/gate_up_sweep_manifest.json"
PEAK_GBS = 288.0  # Wormhole B0 DRAM
BFP8_BYTES_PER_ELEM = 1.0625  # bfloat8_b: 1 mantissa byte + 1 exponent byte per 16


def main() -> int:
    with open(MANIFEST) as f:
        man = json.load(f)
    arms = man["arms"]
    K, n_padded = man["K"], man["n_padded"]
    weight_bytes = K * n_padded * BFP8_BYTES_PER_ELEM

    csvs = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime)
    if not csvs:
        print("no profiler CSV found — did you run under `python -m tracy -p`?")
        return 1
    path = csvs[-1]
    age = os.path.getmtime(path)
    if age < os.path.getmtime(MANIFEST) - 5:
        print(f"REFUSING: newest CSV {path} is OLDER than the manifest.")
        print("The profiler wrote nothing this run. Re-run under `python -m tracy -p -v -r`.")
        return 1
    print(f"csv: {path}")

    # On a multi-chip mesh every op appears once per device; keep a single device so
    # the row order still matches the manifest one-to-one.
    per_dev = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if "matmul" not in (r.get("OP CODE") or "").lower():
                continue
            dur = (r.get("DEVICE KERNEL DURATION [ns]") or "").strip()
            if not dur.isdigit():
                continue
            dev = (r.get("DEVICE ID") or "0").strip()
            per_dev.setdefault(dev, []).append((int(dur), int(float(r.get("CORE COUNT") or 0))))
    if len(per_dev) > 1:
        print(f"devices in csv: {sorted(per_dev)} -> using {sorted(per_dev)[0]}")
    rows = per_dev[sorted(per_dev)[0]] if per_dev else []

    # Each arm emits 1 untimed probe launch (compile + program-cache fill) followed by
    # `reps` steady-state launches, in manifest order.
    need = sum(1 + a["reps"] for a in arms)
    print(f"matmul rows in csv: {len(rows)}, manifest expects {need} (1 probe + {arms[0]['reps']} reps per arm)")
    if len(rows) < need:
        print("REFUSING: fewer matmul rows than the manifest expects — cannot attribute safely.")
        return 1
    rows = rows[-need:]  # this run's matmuls are the trailing block

    print(f"\n### Talker gate/up  M={man['M']} K={K} N={man['N']} (pad {n_padded}), bfloat8_b")
    print(f"    weight = {weight_bytes/1e6:.2f} MB per matmul; peak {PEAK_GBS:.0f} GB/s\n")
    print("| arm | cores | grid | in0_block_w | per_core_N | median us | GB/s | % peak |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")

    out, i, bad = [], 0, 0
    for a in arms:
        chunk = rows[i : i + 1 + a["reps"]]
        i += 1 + a["reps"]
        got = {c for _, c in chunk}
        if a["arm"] == "1d" and got != {a["cores"]}:
            # DRAM-sharded matmuls report the DRAM bank count, not the in0 grid, so
            # only the 1D arm's CORE COUNT is expected to equal `cores`.
            print(f"  !! misalignment: {a['arm']} c{a['cores']} expected CORE COUNT {a['cores']}, csv has {got}")
            bad += 1
        us = [d / 1000.0 for d, _ in chunk[1:]]  # drop the probe
        med = statistics.median(us)
        gbs = weight_bytes / (med * 1e-6) / 1e9
        tag = "  <-- SHIPPED" if a["shipped"] else ""
        print(
            f"| {a['arm']} | {a['cores']} | {a['grid']} | {a['in0_block_w']} | {a['per_core_N']} "
            f"| {med:.1f} | {gbs:.0f} | {100*gbs/PEAK_GBS:.1f}{tag} |"
        )
        out.append((a, med, gbs))

    if bad:
        print(f"\n!! {bad} arm(s) misaligned — treat the table as unreliable and re-run.")
    base = next((o for o in out if o[0]["shipped"]), None)
    best = min(out, key=lambda o: o[1])
    print("\n--- summary ---")
    if base:
        print(
            f"shipped : {base[0]['arm']} {base[0]['cores']} cores -> {base[1]:.1f} us, {base[2]:.0f} GB/s "
            f"({100*base[2]/PEAK_GBS:.1f} % of peak)"
        )
    print(
        f"best    : {best[0]['arm']} {best[0]['cores']} cores ({best[0]['grid']}) -> "
        f"{best[1]:.1f} us, {best[2]:.0f} GB/s ({100*best[2]/PEAK_GBS:.1f} % of peak)"
    )
    if base:
        d = base[1] - best[1]
        if d > 0:
            print(
                f"delta   : -{d:.1f} us per matmul -> gate+up -{2*d:.1f} us/layer, "
                f"-{2*d*28/1000:.2f} ms over 28 Talker layers"
            )
        else:
            print("delta   : the shipped arm is already the best in this sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
