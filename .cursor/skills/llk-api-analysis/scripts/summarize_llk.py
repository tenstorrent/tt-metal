#!/usr/bin/env python3
"""Summary stats for a per-call LLK CSV: config count + data-format / fidelity /
tile-dim / dst-accum distributions. Reads the columns produced by
`llk_api_analyzer -f csv`.

Usage: python3 summarize_llk.py <per_call.csv>
"""
import collections
import csv
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "llk_report.csv"


def _values(cell):
    """'cb0=Float16_b, cb1=Bfp8_b' -> ['Float16_b','Bfp8_b']; '-'/'' -> []."""
    if cell.strip() in ("", "-"):
        return []
    out = []
    for part in cell.split(","):
        part = part.strip()
        if part:
            out.append(part.split("=", 1)[1] if "=" in part else part)
    return out


def main():
    fmt = collections.Counter()
    tile = collections.Counter()
    fidelity = collections.Counter()
    dst_accum = collections.Counter()
    n = 0
    with open(SRC, newline="") as f:
        for r in csv.DictReader(f):
            n += 1
            for col in ("Input Data Formats", "Output Data Formats"):
                fmt.update(_values(r.get(col, "")))
            tile.update(_values(r.get("Tile Dims", "")))
            fidelity.update(_values(r.get("Math Fidelity", "")))
            dst_accum.update(_values(r.get("FP32 Dest Accum", "")))

    print(f"per-call configs: {n}\n")
    for title, counter in (
        ("Data formats (in+out CB refs)", fmt),
        ("Tile dims", tile),
        ("Math fidelity", fidelity),
        ("FP32 dest accum", dst_accum),
    ):
        print(f"=== {title} ===")
        for k, v in counter.most_common():
            print(f"  {k:16} {v}")
        print()


if __name__ == "__main__":
    main()
