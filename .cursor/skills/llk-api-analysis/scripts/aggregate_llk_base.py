#!/usr/bin/env python3
"""Collapse the per-API CSV further: group by *base* LLK API name (template
parameters stripped) and list each distinct template-parameter set as a
bracketed entry in 'Op Args'."""
import csv
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/localdev/rtawfik/llk_report_by_api.csv"
DST = sys.argv[2] if len(sys.argv) > 2 else "/localdev/rtawfik/llk_report_by_base_api.csv"


def base_and_template(api):
    s = api.strip().strip("`").strip()
    if "<" in s:
        return s[: s.index("<")], s[s.index("<") + 1 : s.rindex(">")]
    return s, ""


def split_list(cell):
    return [] if cell.strip() in ("", "-") else [t.strip() for t in cell.split(",") if t.strip()]


def join_uniq(values, sep=", "):
    seen = {}
    for v in values:
        seen.setdefault(v, None)
    return sep.join(seen) if seen else "-"


with open(SRC, newline="") as f:
    rows = list(csv.DictReader(f))

groups = {}
order = []
for r in rows:
    base, tmpl = base_and_template(r["LLK API"])
    if base not in groups:
        groups[base] = {
            k: []
            for k in (
                "perms",
                "TTNN Ops",
                "Input Data Formats",
                "Output Data Formats",
                "Tile Dims",
                "Math Fidelity",
                "Math Approx",
                "FP32 Dest Accum",
                "Dst Sync Mode",
            )
        }
        order.append(base)
    g = groups[base]
    g["perms"].append(tmpl)
    for op in split_list(r["TTNN Ops"]):
        g["TTNN Ops"].append(op)
    for col in (
        "Input Data Formats",
        "Output Data Formats",
        "Tile Dims",
        "Math Fidelity",
        "Math Approx",
        "FP32 Dest Accum",
        "Dst Sync Mode",
    ):
        g[col] += split_list(r[col])

cols = [
    "LLK API",
    "Number of Configs",
    "TTNN Ops",
    "Op Args",
    "Input Data Formats",
    "Output Data Formats",
    "Tile Dims",
    "Math Fidelity",
    "Math Approx",
    "FP32 Dest Accum",
    "Dst Sync Mode",
]

with open(DST, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for base in sorted(order):
        g = groups[base]
        # ordered-unique template permutations
        perms, seen = [], set()
        for p in g["perms"]:
            if p not in seen:
                seen.add(p)
                perms.append(p)
        nonempty = [p for p in perms if p]
        op_args = ", ".join(f"[{p}]" for p in nonempty) if nonempty else "-"
        w.writerow(
            [
                base,
                len(perms),
                join_uniq(g["TTNN Ops"]),
                op_args,
                join_uniq(g["Input Data Formats"]),
                join_uniq(g["Output Data Formats"]),
                join_uniq(g["Tile Dims"]),
                join_uniq(g["Math Fidelity"]),
                join_uniq(g["Math Approx"]),
                join_uniq(g["FP32 Dest Accum"]),
                join_uniq(g["Dst Sync Mode"]),
            ]
        )

print(f"per-API rows in       : {len(rows)}")
print(f"distinct base APIs out: {len(order)}")
print(f"wrote                 : {DST}")
