#!/usr/bin/env python3
"""Collapse the per-config LLK CSV into one row per LLK API."""
import csv
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/localdev/rtawfik/llk_report.csv"
DST = sys.argv[2] if len(sys.argv) > 2 else "/localdev/rtawfik/llk_report_by_api.csv"

BACKTICK = re.compile(r"`[^`]*`")


def split_ops(cell):
    """TTNN ops are backtick-quoted tokens; pull them out individually."""
    toks = BACKTICK.findall(cell)
    return toks if toks else ([] if cell.strip() in ("", "-") else [cell.strip()])


def split_pairs_value(cell):
    """Cells like 'cb0=Float16_b, cb1=Bfp8_b' -> ['Float16_b', 'Bfp8_b'].

    Falls back to the raw comma-split token when there is no '=' (keeps it robust).
    """
    if cell.strip() in ("", "-"):
        return []
    out = []
    for part in cell.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(part.split("=", 1)[1] if "=" in part else part)
    return out


def split_scalar(cell):
    return [] if cell.strip() in ("", "-") else [cell.strip()]


def split_opargs(cell):
    """Keep whole arg-combos distinct (they may contain '='); '-' means none."""
    return [] if cell.strip() in ("", "-") else [cell.strip()]


# ordered-unique join
def join_uniq(values, sep=", "):
    seen = {}
    for v in values:
        if v not in seen:
            seen[v] = None
    return sep.join(seen) if seen else "-"


with open(SRC, newline="") as f:
    rows = list(csv.DictReader(f))

groups = {}
order = []
for r in rows:
    api = r["LLK API"]
    if api not in groups:
        groups[api] = {
            "count": 0,
            "TTNN Ops": [],
            "Op Args": [],
            "Input Data Formats": [],
            "Output Data Formats": [],
            "Tile Dims": [],
            "Math Fidelity": [],
            "Math Approx": [],
            "FP32 Dest Accum": [],
            "Dst Sync Mode": [],
        }
        order.append(api)
    g = groups[api]
    g["count"] += 1
    g["TTNN Ops"] += split_ops(r["TTNN Ops"])
    g["Op Args"] += split_opargs(r["Op Args"])
    g["Input Data Formats"] += split_pairs_value(r["Input Data Formats"])
    g["Output Data Formats"] += split_pairs_value(r["Output Data Formats"])
    g["Tile Dims"] += split_pairs_value(r["Tile Dims"])
    g["Math Fidelity"] += split_scalar(r["Math Fidelity"])
    g["Math Approx"] += split_scalar(r["Math Approx"])
    g["FP32 Dest Accum"] += split_scalar(r["FP32 Dest Accum"])
    g["Dst Sync Mode"] += split_scalar(r["Dst Sync Mode"])

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
    for api in sorted(order):
        g = groups[api]
        w.writerow(
            [
                api,
                g["count"],
                join_uniq(g["TTNN Ops"]),
                join_uniq(g["Op Args"], sep=" | "),
                join_uniq(g["Input Data Formats"]),
                join_uniq(g["Output Data Formats"]),
                join_uniq(g["Tile Dims"]),
                join_uniq(g["Math Fidelity"]),
                join_uniq(g["Math Approx"]),
                join_uniq(g["FP32 Dest Accum"]),
                join_uniq(g["Dst Sync Mode"]),
            ]
        )

print(f"input rows (configs): {len(rows)}")
print(f"distinct LLK APIs    : {len(order)}")
print(f"wrote                : {DST}")
