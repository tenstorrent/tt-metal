#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Split ag_kv of a TP=2 (profile_4x2.py) zone profile into the harness's per-head copies and the rest.

  ag_kv_split.py --per-op per_op_4x2.csv --run-id p4x2_w4096_h141312_prose --per-device profiles/<run>/per_device.json

Appends per sparse layer, with the same key columns as that run's ag_kv row:
  ag_kv_head_copies  = attn/ag_kv/head_slice + attn/ag_kv/head_concat (per chip, then worst/mean/min)
  ag_kv_native_est   = ag_kv - ag_kv_head_copies (the gathers alone), with ag_kv's roofline
Skips a run whose rows are already there.
"""

import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-op", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--per-device", required=True)
    a = ap.parse_args()
    import json

    pd = json.load(open(a.per_device))
    rows = list(csv.DictReader(open(a.per_op)))
    cols = list(rows[0].keys())
    if any(r["run_id"] == a.run_id and r["op"].startswith("ag_kv_") for r in rows):
        print(f"[ag_kv_split] {a.run_id}: rows already present")
        return
    lt = {r["layer"]: float(r["worst_ms"]) for r in rows if r["run_id"] == a.run_id and r["op"] == "layer_total"}
    out = []
    for r in rows:
        if r["run_id"] != a.run_id or r["op"] != "ag_kv":
            continue
        base = f"profiled_chunk/layer{int(r['layer']):02d}_sparse/attn/ag_kv"
        agkv = pd.get(base, {})
        subs = [pd[p] for p in (f"{base}/head_slice", f"{base}/head_concat") if p in pd]
        devs = sorted(agkv)
        copies = {d: sum(s.get(d, 0.0) for s in subs) for d in devs}
        native = {d: agkv[d] - copies[d] for d in devs}
        roof = float(r["roof_ms"]) if r["roof_ms"] else None
        for op, v, rf in (("ag_kv_head_copies", copies, None), ("ag_kv_native_est", native, roof)):
            w, m, n = max(v.values()), sum(v.values()) / len(v), min(v.values())
            row = dict(r)
            row.update(
                op=op,
                calls="",
                worst_ms=round(w, 4),
                mean_ms=round(m, 4),
                min_ms=round(n, 4),
                roof_ms="" if rf is None else round(rf, 4),
                eff_worst="" if rf is None or w <= 0 else round(rf / w, 4),
                eff_mean="" if rf is None or m <= 0 else round(rf / m, 4),
                share_of_layer_worst=round(w / lt[r["layer"]], 4),
            )
            out.append(row)
    with open(a.per_op, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writerows(out)
    print(f"[ag_kv_split] {a.run_id}: {len(out)} rows")


if __name__ == "__main__":
    main()
