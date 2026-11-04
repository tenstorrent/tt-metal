# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Table of a test_matmul_reduce_scatter_model_compare_bh.py run: one row per case, device time per path.

  python3 tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/model_compare/summarize.py generated/mrs_model_compare.jsonl
      [--wall] [--csv out.csv]

The last record of a (mesh, case, path) wins, so a re-run of some cases overrides an earlier run in the same file.
ours/prod is prod time / ours time (> 1: ours faster).
"""

import argparse
import csv
import json
import math
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--wall", action="store_true", help="wall time per replay instead of profiler device time")
    ap.add_argument("--csv")
    args = ap.parse_args()
    key = "wall_ns" if args.wall else "device_ns"

    recs = {}
    for line in open(args.jsonl):
        r = json.loads(line)
        recs[(r.get("mesh"), r["case"], r["path"])] = r
    cases = {}
    for (mesh, case, path), r in recs.items():
        cases.setdefault((mesh, case), {})[path] = r

    def cell(r):
        if r is None:
            return "-"
        if r["status"] == "ok":
            return f"{r[key] / 1e3:.1f}"
        return {"error": "ERR", "wrong_output": "WRONG", "n/a": "n/a"}.get(r["status"], r["status"])

    rows, ratios = [], []
    for (mesh, case), paths in cases.items():
        ours = paths.get("ours")
        prod = next((r for r in paths.values() if r["role"] == "prod"), None)
        alts = [r for r in paths.values() if r["role"] == "alt"]
        ratio = None
        if ours and prod and ours["status"] == "ok" and prod["status"] == "ok":
            ratio = prod[key] / ours[key]
            ratios.append(ratio)
        any_r = next(iter(paths.values()))
        rows.append(
            {
                "mesh": mesh,
                "case": case,
                "model": any_r["model"],
                "M": any_r["M"],
                "K": any_r["K"],
                "N": any_r["N"],
                "prod_path": prod["path"] if prod else "-",
                "prod_us": cell(prod),
                "alts": " ".join(f"{a['path']}={cell(a)}" for a in alts) or "-",
                "ours_us": cell(ours),
                "ours/prod": f"{ratio:.2f}x" if ratio else "-",
                "notes": "; ".join(
                    x
                    for x in (
                        (prod or {}).get("info", {}).get("adapted")
                        if isinstance((prod or {}).get("info"), dict)
                        else None,
                        *(f"{r['path']}: {r['error'][:90]}" for r in paths.values() if r.get("error")),
                    )
                    if x
                ),
            }
        )

    cols = ["mesh", "case", "model", "M", "K", "N", "prod_path", "prod_us", "alts", "ours_us", "ours/prod", "notes"]
    width = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols if c != "notes"}
    print("  ".join(c.ljust(width[c]) for c in cols if c != "notes") + "  notes")
    for r in rows:
        print("  ".join(str(r[c]).ljust(width[c]) for c in cols if c != "notes") + "  " + r["notes"])
    if ratios:
        geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
        print(
            f"\n{len(ratios)} cases with both ok: geomean prod/ours {geo:.2f}x, "
            f"min {min(ratios):.2f}x, max {max(ratios):.2f}x ({'wall' if args.wall else 'device'} time)"
        )
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
