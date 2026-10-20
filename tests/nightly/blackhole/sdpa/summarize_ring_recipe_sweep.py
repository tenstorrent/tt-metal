# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize ring_sdpa_recipe_plugin results: one row per test, one column group per precision.

    python tests/nightly/blackhole/sdpa/summarize_ring_recipe_sweep.py [generated/sdpa_recipe_sweep] > table.md

Per test and precision: outcome, the worst PCC and relative L2 the test computed, and its utilization or device
duration when it profiled. Then failures grouped by their error line, per precision.
"""
import collections
import json
import os
import sys

ORDER = ["LEGACY", "STANDARD", "BALANCED", "ACCURATE", "FAST"]


def load(directory):
    results = collections.defaultdict(dict)
    for precision in ORDER:
        path = os.path.join(directory, f"ring_joint_{precision.lower()}.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path):
            row = json.loads(line)
            results[row["nodeid"].split("::", 1)[1]][precision] = row
    return results


def cell(row):
    if row is None:
        return "–"
    mark = {"passed": "✓", "failed": "✗", "skipped": "skip"}.get(row["outcome"], row["outcome"])
    parts = [mark]
    metrics = [m for m in row["metrics"] if m.get("rel_l2_pct") is not None]
    if metrics:
        worst = max(metrics, key=lambda m: m["rel_l2_pct"])
        parts.append(f"{worst['rel_l2_pct']:.2f}% / {worst['pcc']:.5f}")
    utils = [p["util_pct"] for p in row["perf"] if "util_pct" in p]
    durations = [p["duration_ms"] for p in row["perf"] if "duration_ms" in p]
    if utils:
        parts.append(f"{utils[-1]:.1f}%u")
    elif durations:
        parts.append(f"{durations[-1]:.3f}ms")
    return " ".join(parts)


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "generated/sdpa_recipe_sweep"
    results = load(directory)
    present = [p for p in ORDER if any(p in r for r in results.values())]
    print("| test | " + " | ".join(present) + " |")
    print("|---|" + "---|" * len(present))
    for test in results:
        print(f"| {test} | " + " | ".join(cell(results[test].get(p)) for p in present) + " |")
    print("\ncell: outcome, worst rel L2 / its PCC, utilization (u) or device time\n")
    for precision in present:
        counts = collections.Counter(r[precision]["outcome"] for r in results.values() if precision in r)
        print(f"### {precision}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        errors = collections.Counter(
            r[precision]["error"][:160]
            for r in results.values()
            if precision in r and r[precision]["outcome"] == "failed"
        )
        for error, n in errors.most_common():
            print(f"- {n}× {error}")
        print()


if __name__ == "__main__":
    main()
