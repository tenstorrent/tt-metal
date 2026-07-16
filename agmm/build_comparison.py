# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Join fused AGMM vs isolated (AG + MM) measurements into one comparison table.

Reads three result CSVs produced by run_sweeps:
  - agmm/sweep_latest.csv          fused all_gather_minimal_matmul_async (per shape)
  - agmm/isolated_mm_latest.csv    isolated minimal_matmul       (per shape, id + "_mm")
  - agmm/isolated_ag_latest.csv    isolated all_gather_async     (deduped by device/M/K)

For each AGMM shape: agmm_us, mm_us, ag_us, serial = ag+mm, and the fusion win
(how much the fused op saves vs running the two ops back-to-back, and the implied
overlap = agmm / serial). Writes agmm/comparison.csv and prints a table.
"""

import csv
import os
import sys

_D = os.path.dirname(os.path.abspath(__file__))


def _load(path):
    if not os.path.exists(path):
        return {}
    return {r["shape_id"]: r for r in csv.DictReader(open(path))}


def _us(row):
    if row and row.get("status") == "OK" and row.get("best_duration_us"):
        return float(row["best_duration_us"])
    return None


def build():
    agmm = _load(os.path.join(_D, "sweep_latest.csv"))
    mm = _load(os.path.join(_D, "isolated_mm_latest.csv"))
    ag = _load(os.path.join(_D, "isolated_ag_latest.csv"))

    rows = []
    for sid, a in agmm.items():
        if a.get("op_type") != "agmm":
            continue
        dev, M, K = a["device_config"], a["M"], a["K"]
        ag_id = f"ag_{dev}_m{M}_k{K}"
        agmm_us = _us(a)
        mm_us = _us(mm.get(sid + "_mm"))
        ag_us = _us(ag.get(ag_id))
        serial = (ag_us + mm_us) if (ag_us is not None and mm_us is not None) else None
        save_us = (serial - agmm_us) if (serial is not None and agmm_us is not None) else None
        save_pct = (100.0 * save_us / serial) if (save_us is not None and serial) else None
        overlap = (agmm_us / serial) if (serial and agmm_us is not None) else None
        rows.append(
            {
                "shape_id": sid,
                "device_config": dev,
                "M": int(M),
                "K": int(K),
                "N": int(a["N"]),
                "fusion": a.get("fusion_summary", "-"),
                "agmm_us": agmm_us,
                "ag_us": ag_us,
                "mm_us": mm_us,
                "serial_us": serial,
                "fusion_save_us": save_us,
                "fusion_save_pct": save_pct,
                "overlap_ratio": overlap,
            }
        )
    rows.sort(key=lambda r: (r["device_config"], r["M"], r["K"], r["N"]))

    out = os.path.join(_D, "comparison.csv")
    cols = list(rows[0].keys()) if rows else []
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else (round(v, 1) if isinstance(v, float) else v)) for k, v in r.items()})

    def fmt(v, s="%.1f"):
        return "  -  " if v is None else s % v

    print(
        f"\n{'shape_id':<28}{'M':>6}{'K':>7}{'N':>6}  {'AGMM':>7}{'AG':>7}{'MM':>7}{'AG+MM':>7}{'save%':>7}{'ovlp':>6}"
    )
    print("-" * 100)
    n_ok = 0
    for r in rows:
        if r["agmm_us"] and r["serial_us"]:
            n_ok += 1
        print(
            f"{r['shape_id']:<28}{r['M']:>6}{r['K']:>7}{r['N']:>6}  "
            f"{fmt(r['agmm_us']):>7}{fmt(r['ag_us']):>7}{fmt(r['mm_us']):>7}{fmt(r['serial_us']):>7}"
            f"{fmt(r['fusion_save_pct']):>7}{fmt(r['overlap_ratio'], '%.2f'):>6}"
        )
    print("-" * 100)
    print(f"{n_ok}/{len(rows)} shapes have all three measurements. Wrote {out}")
    return rows


if __name__ == "__main__":
    build()
