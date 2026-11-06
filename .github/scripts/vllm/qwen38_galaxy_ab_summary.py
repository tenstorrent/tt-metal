#!/usr/bin/env python3
"""Markdown summary of a same-machine A1 B1 A2 B2 run of qwen38_galaxy_ab_same_machine.sh."""

import json
import pathlib
import statistics
import sys

TAGS = ["A1", "B1", "A2", "B2"]


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def p50(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, scale=1.0, nd=1):
    return "n/a" if x is None else f"{x * scale:.{nd}f}"


def same(a, b, key):
    if not a or not b:
        return "n/a"
    xs, ys = a[key], b[key]
    if isinstance(xs, dict):
        return "yes" if xs["text"] == ys["text"] else "no"
    if xs and isinstance(xs[0], list):
        xs, ys = [t for s in xs for t in s], [t for s in ys for t in s]
    return f"{sum(x['text'] == y['text'] for x, y in zip(xs, ys))}/{len(xs)}"


def hit_rate(c):
    return c["hits"] / c["queried"] if c and c.get("queried") else None


def main():
    out = pathlib.Path(sys.argv[1])
    startup = {}
    for line in (out / "ab_startup.txt").read_text().splitlines() if (out / "ab_startup.txt").exists() else []:
        tag, _, secs = line.split()
        startup[tag] = int(secs)
    g = {t: load(out / f"ab_{t}_greedy.json") for t in TAGS}
    rows = [
        ("Server start, s", lambda t: str(startup.get(t, "n/a"))),
    ]
    for c in (32, 8):
        b = {t: load(out / f"ab_{t}_bench_c{c}.json") for t in TAGS}
        rows += [
            (f"Bench c{c}: output tok/s", lambda t, b=b: fmt(b[t] and b[t].get("output_throughput"))),
            (f"Bench c{c}: TPOT p50, ms", lambda t, b=b: fmt(b[t] and b[t].get("median_tpot_ms"), nd=2)),
            (f"Bench c{c}: TTFT p50, ms", lambda t, b=b: fmt(b[t] and b[t].get("median_ttft_ms"), nd=0)),
            (f"Bench c{c}: TTFT p99, ms", lambda t, b=b: fmt(b[t] and b[t].get("p99_ttft_ms"), nd=0)),
        ]

    def gfield(fn):
        return lambda t: fn(g[t]) if g[t] else "n/a"

    rows += [
        ("Single prompts one at a time: TTFT p50, s", gfield(lambda d: fmt(p50(r["ttft"] for r in d["b1"]), nd=3))),
        ("Follow-ups: TTFT p50, s", gfield(lambda d: fmt(p50(r["ttft"] for r in d["followup"]), nd=3))),
        ("Follow-ups: cache hit rate", gfield(lambda d: fmt(hit_rate(d["followup_cache"]), nd=3))),
        ("Sessions: follow-up TTFT p50, s", gfield(lambda d: fmt(p50(t["ttft"] for s in d["sessions"] for t in s[1:]), nd=3))),
        ("Sessions: TPOT p50, ms", gfield(lambda d: fmt(p50(t["tpot"] for s in d["sessions"] for t in s), 1000, 2))),
        ("Sessions: cache hit rate", gfield(lambda d: fmt(hit_rate(d["sessions_cache"]), nd=3))),
        ("Sessions: wall, s", gfield(lambda d: fmt(d["sessions_wall"]))),
        ("124k-token prompt: TTFT, s", gfield(lambda d: fmt(d["long"]["ttft"], nd=2))),
    ]
    print("## Same-machine A/B: A = model-owned GDN state (block 64), B = QWEN36_MAMBA_SPEC=1 (block 1024)\n")
    print("| Measurement | " + " | ".join(TAGS) + " |")
    print("| --- |" + " --- |" * len(TAGS))
    for name, f in rows:
        print(f"| {name} | " + " | ".join(f(t) for t in TAGS) + " |")

    print("\n### Greedy replies equal between two servers\n")
    pairs = [("A1", "A2"), ("B1", "B2"), ("A1", "B1"), ("A2", "B2")]
    keys = ["b1", "b32", "followup", "sessions", "long", "image"]
    print("| Pair | " + " | ".join(keys) + " |")
    print("| --- |" + " --- |" * len(keys))
    for x, y in pairs:
        print(f"| {x} vs {y} | " + " | ".join(same(g[x], g[y], k) for k in keys) + " |")


if __name__ == "__main__":
    main()
