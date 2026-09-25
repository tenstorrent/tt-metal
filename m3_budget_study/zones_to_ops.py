#!/usr/bin/env python3
"""Rebuild results/ops.csv from every zone-profile log (e3*_h<h>_n<n>.log) and its parsed zones.json.

Each profile log names its capture ("CSV: <dir>/ops_perf_results_*.csv"); parse_zone_perf.py --json
must already have written <dir>/zones.json. One row per (run, layer, zone).

  zones_to_ops.py results
"""
import csv, glob, json, os, re, sys

COLS = ["run_id", "layer", "layer_type", "zone", "device_ms_worst_chip", "device_ms_mean", "ccl_bytes", "notes"]
CCL = ("allgather", "all_gather", "reduce_scatter", "ag_", "dispatch", "combine", "allreduce")


def main(res):
    rows = []
    for log in sorted(glob.glob(os.path.join(res, "logs", "e3*_h*_n*.log"))):
        run_id = os.path.basename(log)[:-4]
        csvs = re.findall(r"CSV: (\S+)", open(log, errors="replace").read())
        if not csvs:
            continue
        zj = os.path.join(os.path.dirname(csvs[-1]), "zones.json")
        if not os.path.exists(zj):
            print(f"[ops] {run_id}: no zones.json next to {csvs[-1]}")
            continue
        note = "zeroed cache (SKIP_PREFIX)" if "SKIP_PREFIX=1" in open(log, errors="replace").read() else "real history"
        for path, z in json.load(open(zj))["zones"].items():
            parts = path.split("/")
            if len(parts) < 2:
                layer, ltype, zone = "", "", "profiled_chunk"
            else:
                m = re.match(r"layer(\d+)_(\w+)", parts[1])
                if not m:
                    continue
                layer, ltype, zone = int(m.group(1)), m.group(2), "/".join(parts[2:]) or "(layer total)"
            ccl = int(z["mib"] * 2**20) if any(k in zone for k in CCL) else ""
            rows.append(
                dict(
                    run_id=run_id,
                    layer=layer,
                    layer_type=ltype,
                    zone=zone,
                    device_ms_worst_chip=round(z["ms_max"], 4),
                    device_ms_mean=round(z["ms_mean"], 4),
                    ccl_bytes=ccl,
                    notes=note,
                )
            )
    with open(os.path.join(res, "ops.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[ops] wrote {len(rows)} rows")


if __name__ == "__main__":
    main(sys.argv[1])
