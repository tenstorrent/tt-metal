#!/usr/bin/env python3
"""Report the NOP alignment sweep: does code layout toggle the timing bistability?

Each pass puts N nops before the pack loop and changes nothing else.  N nops cost
N cycles for the whole run, so a change in the number of states is a layout effect
rather than a timing effect.

Usage:  perf_nop_alignment_report.py [dir]        (default ~/nopsweep)
"""
import glob
import re
import sys
from pathlib import Path

import pandas as pd

# Two measurements belong to the same state when they sit within this fraction of
# the median; ordinary jitter inside one state is about 0.5%.
STATE_TOL = 0.008


def per_run_total(path):
    """L1_TO_L1 duration for each run: unpack TILE_LOOP start -> pack TILE_LOOP end."""
    d = pd.read_csv(path)
    d = d[d.marker == "TILE_LOOP"]
    end = d[(d.thread == "pack") & (d.type == "ZONE_END")].sort_values("run_index")
    start = d[(d.thread == "unpack") & (d.type == "ZONE_START")].sort_values(
        "run_index"
    )
    return pd.Series(
        end.timestamp.values - start.timestamp.values, index=end.run_index.values
    ).sort_index()


def states(s):
    """Cluster measurements into states: (mean cycles, run count), fastest first."""
    tol = s.median() * STATE_TOL
    groups = []
    for v in sorted(s):
        if groups and v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [(sum(g) / len(g), len(g)) for g in groups]


def text_size(root, name):
    """Pack ELF text size from the perf CSV, to confirm the binary really changed."""
    for f in glob.glob(str(root / name / "**" / "*.csv"), recursive=True):
        if f.endswith((".post.csv", ".counters.csv")):
            continue
        t = pd.read_csv(f, low_memory=False)
        for col in t.columns:
            if "TEXT_SIZE" in col:
                return int(t[col].dropna().iloc[0])
    return None


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "nopsweep")
    files = sorted(
        root.glob("nop*_profiler.csv"),
        key=lambda p: int(re.search(r"nop(\d+)_", p.name).group(1)),
    )
    if not files:
        print(f"no profiler dumps found under {root}")
        return 1

    rows = []
    for f in files:
        n = int(re.search(r"nop(\d+)_", f.name).group(1))
        s = per_run_total(f)
        st = states(s)
        med = s.median()
        rows.append(
            {
                "nops": n,
                "runs": len(s),
                "text_size": text_size(root, f"nop{n}"),
                "median": round(med),
                "states": len(st),
                "gap_%": (
                    round((st[-1][0] - st[0][0]) / med * 100, 3) if len(st) > 1 else 0.0
                ),
                "move_%": round((s.max() - s.min()) / med * 100, 3),
                "detail": "  ".join(f"{m:.0f}x{c}" for m, c in st),
            }
        )

    print(pd.DataFrame(rows).to_string(index=False))
    two = [r["nops"] for r in rows if r["states"] > 1]
    one = [r["nops"] for r in rows if r["states"] == 1]
    print(f"\ntwo states at nops: {two}")
    print(f"one state  at nops: {one}")
    print(
        "\nIf both lists are non-empty, code layout toggles the effect and the cause"
        "\nis instruction placement, not added time. If every pass has two states,"
        "\nnops do not disturb it and the timestamps act through something else."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
