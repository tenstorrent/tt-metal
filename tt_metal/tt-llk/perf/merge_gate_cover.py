#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Check, and refresh, the merge gate's node list against the full suite.

The list must hold, for every compile key of the full suite (so every ELF the suite
builds): each value of every text axis, the smallest and largest value of every
numeric axis, and one test per production config. The input is the compile pass run
with LLK_PERF_COVER_MAP=<dir>. Exit 1 when the list misses something; --out then
holds the list plus the fewest tests that close the gap.
"""
import argparse
import collections
import heapq
import json
from pathlib import Path


def number(value):
    try:
        x = float(value)
    except ValueError:
        return None
    return x if x == x else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True, type=Path)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--prod", required=True, type=Path)
    ap.add_argument("--list", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    variants, sweeps = collections.defaultdict(set), {}
    for path in a.map.glob("map.*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            variants[row["nodeid"]].add(row["variant"])
            sweeps[row["nodeid"]] = row["sweep"]

    groups = collections.defaultdict(list)
    for node, vs in variants.items():
        groups[(node.split("[")[0], *sorted(vs))].append(node)
    covers = collections.defaultdict(set)
    for key, nodes in groups.items():
        axes = collections.defaultdict(dict)
        for node in nodes:
            covers[node].add(("compile key", key))
            for axis, value in sweeps[node].items():
                if axis != "speed_of_light" and value not in ("None", "nan"):
                    axes[axis][node] = value
        for axis, values in axes.items():
            nums = {node: number(v) for node, v in values.items()}
            if None in nums.values():
                for node, value in values.items():
                    covers[node].add(("axis value", key, axis, value))
                continue
            low, high = min(nums.values()), max(nums.values())
            for node, x in nums.items():
                if x == low:
                    covers[node].add(("axis value", key, axis, "min"))
                if x == high:
                    covers[node].add(("axis value", key, axis, "max"))

    configs = json.loads(a.prod.read_text())["configs"]
    unmatched = []
    for i, config in enumerate(configs):
        if config["arch"] != a.arch:
            continue
        prefix = config["module"] + ".py::"
        hits = [
            node
            for node, sweep in sweeps.items()
            if node.startswith(prefix)
            and all(sweep.get(k) == v for k, v in config["axes"].items())
        ]
        for node in hits:
            covers[node].add(("production config", i))
        if not hits:
            unmatched.append(config)

    listed = []
    if a.list and a.list.exists():
        listed = [l for l in a.list.read_text().splitlines() if l.strip()]
    chosen = {node for node in listed if node in variants}
    need = set().union(*covers.values())
    for node in chosen:
        need -= covers[node]
    missing = collections.Counter(r[0] for r in need)
    heap = [(-len(covers[n] & need), n) for n in variants if n not in chosen]
    heapq.heapify(heap)
    while need:
        _, node = heapq.heappop(heap)
        gain = len(covers[node] & need)
        if heap and gain < -heap[0][0]:
            heapq.heappush(heap, (-gain, node))
            continue
        chosen.add(node)
        need -= covers[node]

    print(
        json.dumps(
            {
                "arch": a.arch,
                "tests": len(variants),
                "compile keys": len(groups),
                "listed": len(listed),
                "listed, not in the suite": len(listed) - len(set(listed) & set(variants)),
                "missing": dict(missing),
                "production configs without a test": len(unmatched),
                "refreshed list": len(chosen),
            }
        )
    )
    for config in unmatched:
        print(f"No test measures production config {json.dumps(config)}")
    if a.out:
        a.out.write_text("".join(f"{node}\n" for node in sorted(chosen)))
    return 1 if missing or unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
