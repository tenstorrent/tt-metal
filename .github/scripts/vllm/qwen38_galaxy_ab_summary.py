#!/usr/bin/env python3
"""Markdown summary of a same-machine run of qwen38_galaxy_ab_same_machine.sh."""

import json
import pathlib
import re
import statistics
import sys

DEFAULT_TAGS = ["A1", "B1", "A2", "B2"]
# The plugin's cumulative "TT submissions:" line, logged per worker (one per DP rank).
CHAIN_RE = re.compile(
    r"\(Worker pid=(\d+)\).*TT submissions: .*?real drafts: (\d+) valid, (\d+) accepted over (\d+) verify rows; "
    r"chain: (\d+) chained fused steps \([\d.]+% of (\d+) fused verifies\), (\d+) chain breaks(?: \(([^)]*)\))?"
)


STEP_RE = re.compile(r"\[spec step timing\] \d+ steps: inside the model ([\d.]+) ms/step, runner and scheduler ([\d.]+) ms/step")


def step_timing(out, tag):
    """Mean over every logged window of QWEN36_SPEC_STEP_TIMING: (model ms/step, runner and scheduler ms/step)."""
    try:
        text = (out / f"ab_{tag}_server.log").read_text(errors="ignore")
    except OSError:
        return None
    vals = [(float(a), float(b)) for a, b in STEP_RE.findall(text)]
    if not vals:
        return None
    return statistics.mean(v[0] for v in vals), statistics.mean(v[1] for v in vals)


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


def counter(path, name):
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    return sum(float(line.rsplit(" ", 1)[1]) for line in lines if line.startswith(name))


def acceptance(out, tag, c):
    pre, post = out / f"ab_{tag}_metrics_pre_c{c}.txt", out / f"ab_{tag}_metrics_post_c{c}.txt"
    names = ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total")
    vals = [counter(p, n) for p in (pre, post) for n in names]
    if None in vals or vals[3] == vals[1]:
        return None
    return (vals[2] - vals[0]) / (vals[3] - vals[1])


def chain_stats(out, tag):
    """Sum over workers of each worker's last cumulative counter line: (chained share, breaks by reason)."""
    try:
        text = (out / f"ab_{tag}_server.log").read_text(errors="ignore")
    except OSError:
        return None
    last = {}
    for m in CHAIN_RE.finditer(text):
        last[m.group(1)] = m
    if not last:
        return None
    chained = sum(int(m.group(5)) for m in last.values())
    fused = sum(int(m.group(6)) for m in last.values())
    breaks = {}
    for m in last.values():
        for part in (m.group(8) or "").split(", "):
            if "=" in part:
                k, v = part.split("=")
                breaks[k] = breaks.get(k, 0) + int(v)
    return chained / fused if fused else None, breaks


def hit_rate(c):
    return c["hits"] / c["queried"] if c and c.get("queried") else None


def main():
    out = pathlib.Path(sys.argv[1])
    arms_file = out / "ab_arms.txt"
    tags = [t for t in arms_file.read_text().split() if t] if arms_file.exists() else DEFAULT_TAGS
    startup = {}
    for line in (out / "ab_startup.txt").read_text().splitlines() if (out / "ab_startup.txt").exists() else []:
        tag, _, secs = line.split()
        startup[tag] = int(secs)
    g = {t: load(out / f"ab_{t}_greedy.json") for t in tags}
    chains = {t: chain_stats(out, t) for t in tags}
    steps = {t: step_timing(out, t) for t in tags}
    concurrencies = sorted(
        {int(p.name.split("_c")[-1].split(".")[0]) for p in out.glob("ab_*_bench_c*.json")}, reverse=True
    )
    rows = [
        ("Server start, s", lambda t: str(startup.get(t, "n/a"))),
    ]
    for c in concurrencies:
        b = {t: load(out / f"ab_{t}_bench_c{c}.json") for t in tags}
        rows += [
            (f"Bench c{c}: output tok/s", lambda t, b=b: fmt(b[t] and b[t].get("output_throughput"))),
            (f"Bench c{c}: TPOT p50, ms", lambda t, b=b: fmt(b[t] and b[t].get("median_tpot_ms"), nd=2)),
            (f"Bench c{c}: TTFT p50, ms", lambda t, b=b: fmt(b[t] and b[t].get("median_ttft_ms"), nd=0)),
            (f"Bench c{c}: TTFT p99, ms", lambda t, b=b: fmt(b[t] and b[t].get("p99_ttft_ms"), nd=0)),
            (f"Bench c{c}: draft acceptance", lambda t, c=c: fmt(acceptance(out, t, c), nd=3)),
        ]

    def gfield(fn):
        return lambda t: fn(g[t]) if g[t] else "n/a"

    rows += [
        ("Whole server: chained share of fused verifies", lambda t: fmt(chains[t] and chains[t][0], 100.0, 1)),
        (
            "Whole server: chain breaks by reason",
            lambda t: ", ".join(f"{k}={v}" for k, v in sorted(chains[t][1].items())) if chains[t] else "n/a",
        ),
        ("Whole server: host time in the model, ms per spec step", lambda t: fmt(steps[t] and steps[t][0], nd=2)),
        ("Whole server: host time in runner and scheduler, ms per spec step", lambda t: fmt(steps[t] and steps[t][1], nd=2)),
        ("Single prompts one at a time: TTFT p50, s", gfield(lambda d: fmt(p50(r["ttft"] for r in d["b1"]), nd=3))),
        ("Follow-ups: TTFT p50, s", gfield(lambda d: fmt(p50(r["ttft"] for r in d["followup"]), nd=3))),
        ("Follow-ups: cache hit rate", gfield(lambda d: fmt(hit_rate(d["followup_cache"]), nd=3))),
        ("Sessions: follow-up TTFT p50, s", gfield(lambda d: fmt(p50(t["ttft"] for s in d["sessions"] for t in s[1:]), nd=3))),
        ("Sessions: TPOT p50, ms", gfield(lambda d: fmt(p50(t["tpot"] for s in d["sessions"] for t in s), 1000, 2))),
        ("Sessions: cache hit rate", gfield(lambda d: fmt(hit_rate(d["sessions_cache"]), nd=3))),
        ("Sessions: wall, s", gfield(lambda d: fmt(d["sessions_wall"]))),
        ("124k-token prompt: TTFT, s", gfield(lambda d: fmt(d["long"]["ttft"], nd=2))),
    ]
    print("## Same-machine runs: A = model-owned GDN state, B = QWEN36_MAMBA_SPEC=1\n")
    print("| Measurement | " + " | ".join(tags) + " |")
    print("| --- |" + " --- |" * len(tags))
    for name, f in rows:
        print(f"| {name} | " + " | ".join(f(t) for t in tags) + " |")

    greedy = [t for t in tags if g[t]]
    if len(greedy) > 1:
        print("\n### Greedy replies equal to the first arm's\n")
        keys = ["b1", "b32", "followup", "sessions", "long", "image"]
        print("| Pair | " + " | ".join(keys) + " |")
        print("| --- |" + " --- |" * len(keys))
        for y in greedy[1:]:
            print(f"| {greedy[0]} vs {y} | " + " | ".join(same(g[greedy[0]], g[y], k) for k in keys) + " |")


if __name__ == "__main__":
    main()
