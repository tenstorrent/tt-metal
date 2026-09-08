# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Score decode_mm_manifest.json against the newest profiler CSV.

Unlike the gate/up report, every arm here may carry a DIFFERENT weight byte count
(the sweep varies the DRAM pad), so GB/s is computed per arm from its own
``n_padded``. A wider pad therefore cannot buy a better bandwidth number — it has to
win on wall-clock microseconds, which is the only column that reaches the demo.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
import sys

MANIFEST = "generated/decode_mm_manifest.json"
PEAK_GBS = 288.0
BFP8 = 1.0625
LAYERS = 28


def main() -> int:
    with open(MANIFEST) as f:
        man = json.load(f)
    arms = man["arms"]

    csvs = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime)
    if not csvs:
        print("no profiler CSV — run under `python -m tracy -p -v -r`")
        return 1
    path = csvs[-1]
    if os.path.getmtime(path) < os.path.getmtime(MANIFEST) - 5:
        print(f"REFUSING: newest CSV {path} predates the manifest; the profiler wrote nothing.")
        return 1
    print(f"csv: {path}")

    per_dev = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if "matmul" not in (r.get("OP CODE") or "").lower():
                continue
            dur = (r.get("DEVICE KERNEL DURATION [ns]") or "").strip()
            if not dur.isdigit():
                continue
            per_dev.setdefault((r.get("DEVICE ID") or "0").strip(), []).append(int(dur))
    if not per_dev:
        print("no matmul rows in the CSV")
        return 1
    if len(per_dev) > 1:
        print(f"devices: {sorted(per_dev)} -> scoring {sorted(per_dev)[0]}")
    rows = per_dev[sorted(per_dev)[0]]

    need = sum(1 + a["reps"] for a in arms)
    print(f"matmul rows: {len(rows)}, manifest expects {need}")
    if len(rows) < need:
        print("REFUSING: fewer rows than expected — cannot attribute safely.")
        return 1
    rows = rows[-need:]

    out, i = [], 0
    for a in arms:
        chunk = rows[i : i + 1 + a["reps"]]
        i += 1 + a["reps"]
        med = statistics.median(d / 1000.0 for d in chunk[1:])  # drop the probe
        b = a["K"] * a["n_padded"] * BFP8
        out.append((a, med, b / (med * 1e-6) / 1e9))

    for shape in dict.fromkeys(a["shape"] for a, _m, _g in out):
        sub = [o for o in out if o[0]["shape"] == shape]
        a0 = sub[0][0]
        print(f"\n### {shape}  M={man['M']} K={a0['K']} N={a0['N']}, bfloat8_b")
        print("| arm | pad | +bytes | cores | out cores | ibw | median us | GB/s | % peak |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for a, med, gbs in sub:
            pad_pct = 100 * (a["n_padded"] / a["N"] - 1)
            tag = "  <-- SHIPPED" if a["shipped"] else ""
            print(
                f"| {a['label'].split()[0]} | {a['n_padded']} | +{pad_pct:.1f} % | {a['cores']} "
                f"| {a['out_cores']} | {a['in0_block_w']} | {med:.1f} | {gbs:.0f} "
                f"| {100*gbs/PEAK_GBS:.1f}{tag} |"
            )
        base = next((o for o in sub if o[0]["shipped"]), None)
        best = min(sub, key=lambda o: o[1])
        print(f"\n  best   : {best[0]['label']} -> {best[1]:.1f} us ({100*best[2]/PEAK_GBS:.1f} % of peak)")
        if base:
            d = base[1] - best[1]
            n = 2 if shape == "gate_up" else 1
            print(f"  shipped: {base[0]['label']} -> {base[1]:.1f} us ({100*base[2]/PEAK_GBS:.1f} % of peak)")
            print(
                f"  delta  : {-d:+.1f} us/matmul -> {-d*n:+.1f} us/layer, "
                f"{-d*n*LAYERS/1000:+.2f} ms over {LAYERS} layers"
                if d > 0
                else "  delta  : shipped is already best in this sweep."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
