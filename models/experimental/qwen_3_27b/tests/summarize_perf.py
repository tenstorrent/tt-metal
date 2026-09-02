# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Turn a Tracy ops CSV into per-region op tables.

    python models/experimental/qwen_3_27b/tests/summarize_perf.py <profiler_out_dir>

Regions come from signpost pairs and may nest, so a parent's numbers INCLUDE its
children's. Iteration count is inferred from how many times a region appears, and
per-iteration figures are the mean over those. Rows outside any region -- weight
loading, the warmup forward -- are ignored.
"""

import sys
from pathlib import Path

import pandas as pd

DURATION = "DEVICE KERNEL DURATION [ns]"


def regions(df):
    """{name: {depth, start, frames}} from signpost rows. Handles nesting and repeats."""
    stack, out = [], {}
    for i, row in df.iterrows():
        if row.get("OP TYPE") != "signpost":
            continue
        code = str(row["OP CODE"])
        if code.endswith("_START"):
            stack.append((code[: -len("_START")], i))
        elif code.endswith("_END"):
            name = code[: -len("_END")]
            # Match the innermost open region with this name.
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][0] == name:
                    _, start = stack.pop(j)
                    r = out.setdefault(name, {"depth": len(stack), "start": start, "frames": []})
                    r["frames"].append(df.loc[start:i])
                    break
    return out


def _ops(info):
    ops = pd.concat(info["frames"])
    ops = ops[ops["OP TYPE"] != "signpost"].copy()
    ops[DURATION] = pd.to_numeric(ops[DURATION], errors="coerce")
    return ops


def _per_iter(info):
    """(us, dispatches) averaged over the region's occurrences."""
    ops, n = _ops(info), len(info["frames"])
    return ops[DURATION].sum() / n / 1e3, len(ops) / n


def tree(found):
    """Per root region, its instrumented children indented beneath it."""
    roots = sorted((n for n, i in found.items() if i["depth"] == 0), key=lambda n: found[n]["start"])

    for root in roots:
        root_us, root_ops = _per_iter(found[root])
        kids = sorted((n for n in found if n.startswith(root + "_")), key=lambda n: found[n]["start"])

        print(f"\n{'REGION':<40}{'us/iter':>11}{'ops/iter':>10}{'%':>8}{'iters':>7}")
        print(f"{root:<40}{root_us:>11.1f}{root_ops:>10.1f}{100.0:>8.1f}{len(found[root]['frames']):>7}")
        for name in kids:
            info = found[name]
            us, n_ops = _per_iter(info)
            label = "  " * info["depth"] + name[len(root) + 1 :]
            print(f"{label:<40}{us:>11.1f}{n_ops:>10.1f}{100*us/root_us:>8.1f}{len(info['frames']):>7}")

        # Root time outside any instrumented child (out_proj, reshapes, deallocs).
        direct = sum(_per_iter(found[n])[0] for n in kids if found[n]["depth"] == 1)
        rest = root_us - direct
        print(f"{'  [unattributed]':<40}{rest:>11.1f}{'':>10}{100*rest/root_us:>8.1f}")


def per_op(name, info):
    ops, n = _ops(info), len(info["frames"])
    total = ops[DURATION].sum()
    g = ops.groupby("OP CODE")[DURATION].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False)

    print(f"\n--- {name} ---")
    print(f"{'OP CODE':<38}{'n/iter':>9}{'us/iter':>10}{'us/op':>9}{'%':>7}")
    for op, r in g.iterrows():
        print(f"{op:<38}{r['count']/n:>9.1f}{r['sum']/n/1e3:>10.1f}{r['mean']/1e3:>9.1f}{100*r['sum']/total:>7.1f}")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    only = sys.argv[2] if len(sys.argv) > 2 else None  # substring filter on region name

    csvs = sorted(root.glob("reports/*/ops_perf_results_*.csv"))
    if not csvs:
        sys.exit(f"no ops_perf_results CSV under {root}/reports/")

    df = pd.read_csv(csvs[-1], low_memory=False)  # 380+ columns, mixed types
    print(f"{csvs[-1]}  ({len(df)} rows)")

    found = regions(df)
    if not found:
        sys.exit("no signposted regions found -- was the workload run from test_perf.py?")
    if only:
        found = {k: v for k, v in found.items() if only in k}

    tree(found)
    for name, info in sorted(found.items(), key=lambda kv: kv[1]["start"]):
        per_op(name, info)


main()
