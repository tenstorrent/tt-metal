#!/usr/bin/env python3
"""Join regime-A MM (this branch) against main's fused AGMM and the existing unfused AG+MM.

Columns, per LTX/FLUX AGMM shape:
  fused_agmm   main's fused AGMM                      (comparison.csv agmm_us)
  ag           isolated all-gather                    (comparison.csv ag_us)
  mm_old       existing MM                            (comparison.csv mm_us)
  serial_old   ag + mm_old                            = what main-unfused delivers today
  mm_ra        regime-A MM, default picker            (measured here)
  serial_ra    ag + mm_ra                             = the milestone candidate
  vs_fused     serial_ra speedup over main fused AGMM
  vs_serial    serial_ra speedup over ag + mm_old

The AG leg is REUSED, not remeasured: fusing changes only the MM side of the composition, and ag_us was
collected on this same Galaxy. That does mean serial_ra inherits ag_us's collection conditions.

usage: ltxflux_report.py <mm.jsonl> <comparison.csv> [--md]
"""
import csv
import json
import statistics
import sys

MM, CMP = sys.argv[1], sys.argv[2]
AS_MD = "--md" in sys.argv

ra = {}
for line in open(MM):
    r = json.loads(line)
    ra[(r["M"], r["K"], r["N"])] = r

rows = []
for c in csv.DictReader(open(CMP)):
    key = (int(c["M"]), int(c["K"]), int(c["N"]))
    r = ra.get(key)
    if r is None:
        continue

    def f(x):
        return float(x) if x not in ("", None) else None

    ag, mm_old, fused = f(c["ag_us"]), f(c["mm_us"]), f(c["agmm_us"])
    mm_ra = r.get("median_us") if r.get("outcome") == "ok" else None
    rows.append(
        {
            "id": c["shape_id"],
            "fam": "LTX" if c["shape_id"].startswith("ltx") else "FLUX",
            "M": key[0],
            "K": key[1],
            "N": key[2],
            "fusion": c["fusion"],
            "fused": fused,
            "ag": ag,
            "mm_old": mm_old,
            "serial_old": (ag + mm_old) if (ag and mm_old) else None,
            "mm_ra": mm_ra,
            "serial_ra": (ag + mm_ra) if (ag and mm_ra) else None,
            "dram": r.get("dram_pct"),
            "pcc": r.get("pcc"),
            "finite": r.get("finite"),
            "nnf": r.get("n_nonfinite"),
            "spread": r.get("block_medians"),
            "pick": r.get("pick"),
            "err": r.get("err") if r.get("outcome") != "ok" else None,
        }
    )


def ratio(a, b):
    return (a / b) if (a and b) else None


def s(v, p=1):
    return "-" if v is None else f"{v:.{p}f}"


hdr = [
    "shape",
    "fam",
    "fusion",
    "fused_agmm",
    "ag",
    "mm_old",
    "serial_old",
    "mm_ra",
    "serial_ra",
    "vs_fused",
    "vs_serial",
    "dram%",
    "pcc",
    "finite",
    "block medians",
    "pick",
]
out = []
for r in rows:
    vf, vs = ratio(r["fused"], r["serial_ra"]), ratio(r["serial_old"], r["serial_ra"])
    out.append(
        [
            f"{r['M']}x{r['K']}x{r['N']}",
            r["fam"],
            r["fusion"],
            s(r["fused"]),
            s(r["ag"]),
            s(r["mm_old"]),
            s(r["serial_old"]),
            s(r["mm_ra"]) if r["mm_ra"] else f"FAIL",
            s(r["serial_ra"]),
            (f"{vf:.2f}x" if vf else "-"),
            (f"{vs:.2f}x" if vs else "-"),
            s(r["dram"]),
            ("-" if r["pcc"] is None else f"{r['pcc']:.6f}"),
            ("-" if r["finite"] is None else ("yes" if r["finite"] else f"NO({r['nnf']})")),
            ("-" if not r["spread"] else ",".join(f"{x:.1f}" for x in r["spread"])),
            ("-" if not r["pick"] else ",".join(map(str, r["pick"]))),
        ]
    )

if AS_MD:
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    for o in out:
        print("| " + " | ".join(o) + " |")
else:
    w = [max(len(str(x)) for x in [hdr[i]] + [o[i] for o in out]) for i in range(len(hdr))]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    for o in out:
        print("  ".join(str(x).ljust(w[i]) for i, x in enumerate(o)))

ok = [r for r in rows if r["serial_ra"]]
print(f"\n{len(ok)}/{len(rows)} shapes measured; {len(rows)-len(ok)} failed")
for r in rows:
    if not r["mm_ra"]:
        print(f"  FAIL {r['M']}x{r['K']}x{r['N']}: {str(r['err'])[:150]}")
for fam in ("LTX", "FLUX", None):
    g = [r for r in ok if fam is None or r["fam"] == fam]
    if not g:
        continue
    vf = [ratio(r["fused"], r["serial_ra"]) for r in g]
    vs = [ratio(r["serial_old"], r["serial_ra"]) for r in g]
    vf, vs = [x for x in vf if x], [x for x in vs if x]
    tf = sum(r["fused"] for r in g if r["fused"])
    tr = sum(r["serial_ra"] for r in g if r["fused"])
    lbl = fam or "ALL"
    print(
        f"{lbl:5s} n={len(g):2d}  vs_fused median {statistics.median(vf):.2f}x  "
        f"vs_serial_old median {statistics.median(vs):.2f}x  "
        f"| summed fused {tf:.0f}us -> serial_ra {tr:.0f}us = {tf/tr:.2f}x"
    )
bad = [r for r in ok if r["finite"] is False]
lo = [r for r in ok if r["pcc"] is not None and r["pcc"] < 0.99]
print(f"non-finite outputs: {len(bad)}   pcc<0.99: {len(lo)}")
