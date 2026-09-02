#!/usr/bin/env python3
"""Read the probe-ladder profiler dumps and report where the bistability dies.

The ladder instruments a growing number of pack-loop iterations.  Full
instrumentation is known to suppress the two-state behaviour; this report finds
the amount of added work at which that happens, which measures the slack the
mechanism has.

Usage:  perf_probe_ladder_report.py [dir]        (default ~/tsexp3)
"""
import sys
from pathlib import Path

import pandas as pd

PASSES = ("baseline", "probe0", "probe8", "probe32", "ts64")
# Two measurements belong to the same state when they sit within this fraction
# of the median; ordinary jitter inside one state is ~0.5%.
STATE_TOL = 0.008


def per_run_total(path: Path) -> pd.Series:
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


def states(s: pd.Series) -> list[tuple[float, int]]:
    """Cluster the measurements into states: (mean cycles, run count), fastest first."""
    tol = s.median() * STATE_TOL
    groups: list[list[float]] = []
    for v in sorted(s):
        if groups and v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [(sum(g) / len(g), len(g)) for g in groups]


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "tsexp3")
    rows, base = [], None
    for name in PASSES:
        f = root / f"{name}_profiler.csv"
        if not f.exists():
            continue
        s = per_run_total(f)
        st = states(s)
        med = s.median()
        if base is None:
            base = med
        rows.append(
            {
                "pass": name,
                "runs": len(s),
                "median": round(med),
                "overhead": round(med - base),
                "min": round(s.min()),
                "max": round(s.max()),
                "move_%": round((s.max() - s.min()) / med * 100, 3),
                "states": len(st),
                "gap_%": (
                    round((st[-1][0] - st[0][0]) / med * 100, 3) if len(st) > 1 else 0.0
                ),
            }
        )
        print(f"--- {name} ---")
        for mean, n in st:
            print(f"    {mean:9.0f} cycles  x{n:3d} runs")
    if not rows:
        print(f"no profiler dumps found under {root}")
        return 1

    print("\n=== ladder ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nA pass with 2+ states still shows the effect.")
    print("A pass with 1 state has been suppressed by its own instrumentation.")
    print("The slack sits between the overhead of the last 2-state pass and the")
    print("overhead of the first 1-state pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
