# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Score prefill_trio_manifest.json. Each group owns one op code, so the three
sequences align independently and a refused arm in one group cannot shift another."""

from __future__ import annotations

import csv, glob, json, os, statistics, sys

MANIFEST = "generated/prefill_trio_manifest.json"
OPCODE = {
    "rmsnorm": "LayerNormDeviceOperation",
    "silumul": "BinaryNgDeviceOperation",
    "gateup128": "MatmulDeviceOperation",
}
# occurrences per layer in the prefill window
PER_LAYER = {"rmsnorm": 4, "silumul": 1, "gateup128": 2}
LAYERS = 28


def main() -> int:
    man = json.load(open(MANIFEST))
    arms = man["arms"]
    csvs = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime)
    if not csvs:
        print("no profiler CSV")
        return 1
    path = csvs[-1]
    if os.path.getmtime(path) < os.path.getmtime(MANIFEST) - 5:
        print(f"REFUSING: CSV {path} predates the manifest.")
        return 1
    print(f"csv: {path}")

    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            code = (r.get("OP CODE") or "").strip()
            d = (r.get("DEVICE KERNEL DURATION [ns]") or "").strip()
            if code and d.isdigit():
                rows.setdefault(code, []).append((int(d), int(float(r.get("CORE COUNT") or 0))))

    for group in ("rmsnorm", "silumul", "gateup128"):
        sub = [a for a in arms if a["group"] == group]
        if not sub:
            continue
        code = OPCODE[group]
        seq = rows.get(code, [])
        need = sum(1 + a["reps"] for a in sub)
        if len(seq) < need:
            print(f"\n{group}: only {len(seq)} {code} rows, need {need} — skipped")
            continue
        seq = seq[-need:]
        print(f"\n### {group}  ({code}; x{PER_LAYER[group]} per layer, {LAYERS} layers)")
        print("| arm | cores | median us | rel_rms % | us/layer | ms/28 layers |")
        print("|---|---:|---:|---:|---:|---:|")
        i, out = 0, []
        for a in sub:
            n = 1 + a["reps"]
            ch = seq[i : i + n][1:]
            i += n
            med = statistics.median(d / 1000 for d, _c in ch)
            cores = sorted({c for _d, c in ch})
            per = med * PER_LAYER[group]
            rr = "ref" if a.get("rel_rms") is None else f"{a['rel_rms']*100:.4f}"
            out.append((a, med, per))
            print(f"| {a['label']} | {cores} | {med:.1f} | {rr} | {per:.1f} | {per*LAYERS/1000:.2f} |")
        # per-m comparison where the group is m-split
        for mval in sorted({a.get("m") for a, _m, _p in out if a.get("m") is not None}) or [None]:
            grp = [o for o in out if o[0].get("m") == mval] if mval else out
            if not grp:
                continue
            base = next((o for o in grp if o[0].get("rel_rms") is None or o[0].get("shipped")), grp[0])
            best = min(grp, key=lambda o: o[1])
            tag = f"m={mval} " if mval else ""
            d = base[1] - best[1]
            print(
                f"  {tag}shipped {base[0]['label']}: {base[1]:.1f} us -> best {best[0]['label']}: {best[1]:.1f} us"
                f"  ({-d:+.1f} us/op, {-d*PER_LAYER[group]*LAYERS/1000:+.2f} ms/28 layers)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
