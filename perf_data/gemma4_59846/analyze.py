"""Summarize the Gemma4 global SDPA chunk sweep (#59846): best and production tuning per (chunk, prefix)."""

import json
import re
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/data/mmanzoor/sdpa_chunk_sweep/global_sweep.jsonl"
rows = [json.loads(line) for line in open(path)]


def tuning(r):
    return f"q{r['q_chunk']}/k{r['k_chunk']}/ks{r['max_k_splits']}/{'seg' if r['segmented'] else 'unseg'}"


def spread(r):
    return 100 * (r["max_us"] - r["min_us"]) / r["median_us"]


by_point = defaultdict(list)
for r in rows:
    by_point[(r["chunk_size"], r["prefix"], r["final"])].append(r)

print(
    "| C | c | prefix | final | best tuning | median us | spread % | util % | production tuning | prod us | prod util % | ok/err |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for (C, prefix, final), rs in sorted(by_point.items()):
    ok = [r for r in rs if r["status"] == "ok" and "median_us" in r]
    errs = [r for r in rs if r["status"] != "ok"]
    best = min(ok, key=lambda r: r["median_us"]) if ok else None
    prod = next((r for r in ok if r["production"]), None)
    print(
        f"| {C} | {C // 8} | {prefix} | {'y' if final else ''} | "
        + (
            f"{tuning(best)} | {best['median_us']:.1f} | {spread(best):.1f} | {best['util']:.2f} | "
            if best
            else "- | - | - | - | "
        )
        + (f"{tuning(prod)} | {prod['median_us']:.1f} | {prod['util']:.2f} | " if prod else "- | - | - | ")
        + f"{len(ok)}/{len(errs)} |"
    )

print("\nErrors (grouped by tuning, any prefix):")
err = defaultdict(set)
for r in rows:
    if r["status"] != "ok":
        msg = r.get("error") or r["status"]
        m = re.search(r"grow to (\d+) B which is beyond max L1 size", msg)
        reason = f"CBs {int(m.group(1))/1024:.0f} KB > L1 1536 KB" if m else msg.split("\n")[0][-160:]
        err[(r["chunk_size"], r["q_chunk"], r["k_chunk"], reason)].add(r["prefix"])
for (C, q, k, reason), prefixes in sorted(err.items()):
    print(f"- C{C} q{q}/k{k} (all K splits / seg): {reason}; prefixes {sorted(prefixes)}")
