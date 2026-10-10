# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Client for demo/batching_server.py: fire short and long prompts at the prefill server and see how fast they
return.

    python batching_client.py compare                   # two bursts, serial 8k / 4k / 2k chunks then batched: many
                                                        # short prompts, and a 64k prompt with short ones behind it
    python batching_client.py compare 32k 2k 2k 4k      # your own prompts
    python batching_client.py fire 2k 2k 32k 2k         # just fire them (--gap seconds apart, default 0)
    python batching_client.py mode batched|8k|4k|2k     # continuous batching, or serial at that chunk size

Sizes take a k suffix (tokens, rounded up to whole 2k chunks).
"""

import argparse
import json
import threading
import time
import urllib.error
import urllib.request

from rich.console import Console
from rich.table import Table

console = Console()
# compare's default scenarios: everything arrives at once (a burst of load).
SCENARIOS = {
    "many short prompts": ["2k"] * 16,
    "a long prompt with short ones behind it": ["64k"] + ["2k"] * 8,
}


def _post(url, path, payload):
    request = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        return json.loads(error.read())


def _tokens(size):
    size = size.lower()
    return int(float(size[:-1]) * 1024) if size.endswith("k") else int(size)


BASELINES = ("serial 8k", "serial 4k", "serial 2k")  # one request at a time at a fixed chunk size (8k: today's default)


def set_mode(url, mode):
    return _post(url, "/mode", {"mode": mode})["mode"]


def fire(url, sizes, gap):
    """Submit prompts gap seconds apart (one thread each), wait for all; returns (results in order, wall seconds)."""
    results = [None] * len(sizes)

    def run(i, size):
        r = results[i] = _post(url, "/prefill", {"tokens": _tokens(size), "label": f"#{i} {size}"})
        if "error" in r:
            console.print(f"  [red]failed {r.get('label', size)}: {r['error']}")
        else:
            console.print(f"  done  {r['label']:>10}  latency {r['latency_ms'] / 1000:6.2f}s")

    t0 = time.perf_counter()
    threads = []
    for i, size in enumerate(sizes):
        threads.append(threading.Thread(target=run, args=(i, size)))
        threads[-1].start()
        time.sleep(gap)
    for thread in threads:
        thread.join()
    return [r for r in results if r and "error" not in r], time.perf_counter() - t0


def show(title, results, wall):
    if not results:
        return
    longest = max(r["latency_ms"] for r in results)
    table = Table(title=title)
    for col in ("request", "tokens", "queue", "prefill", "latency", ""):
        table.add_column(col)
    for r in results:
        table.add_row(
            r["label"],
            f"{r['tokens']:,}",
            f"{r['queue_ms'] / 1000:.2f}s",
            f"{r['prefill_ms'] / 1000:.2f}s",
            f"{r['latency_ms'] / 1000:.2f}s",
            "█" * max(1, round(40 * r["latency_ms"] / longest)),
        )
    console.print(table)
    tokens = sum(r["tokens"] for r in results)
    console.print(f"{tokens:,} tokens in {wall:.1f}s wall ({tokens / wall:,.0f} tok/s)")


SHORT_MAX = 4096  # prompts up to this many tokens count as short in the summaries


def mean_latency_s(results, short):
    """Mean latency in seconds of the short (<= SHORT_MAX tokens) or long prompts; None if there are none."""
    picked = [r["latency_ms"] for r in results if (r["tokens"] <= SHORT_MAX) == short]
    return sum(picked) / len(picked) / 1000 if picked else None


def latency_change(off, on, short):
    """'a -> b s': mean latency of the short (or long) prompts with batching off, then on; None if there are none."""
    means = [mean_latency_s(results, short) for results in (off, on)]
    return None if None in means else f"{means[0]:.3f}s -> {means[1]:.3f}s"


def compare(url, sizes, gap):
    """The same prompts serial at each baseline chunk size, then batched; returns one summary line per baseline."""
    runs = {}
    for mode in (*BASELINES, "batched"):
        set_mode(url, mode)
        console.print(f"[bold]{mode}[/]: firing {' '.join(sizes)}")
        runs[mode] = fire(url, sizes, gap)
        show(mode, *runs[mode])
    on, on_wall = runs["batched"]
    tokens = sum(r["tokens"] for r in on)
    lines = []
    for mode in BASELINES:
        off, off_wall = runs[mode]
        line = f"vs {mode}: whole mix {off_wall:.2f}s -> {on_wall:.2f}s ({off_wall / on_wall:.2f}x, "
        line += f"{tokens / off_wall:,.0f} -> {tokens / on_wall:,.0f} tok/s)"
        for short, name in ((True, "short (<= 4k)"), (False, "long")):
            if change := latency_change(off, on, short):
                line += f"; {name} mean latency {change}"
        lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 127.0.0.1, not localhost: resolving localhost to IPv6 first costs every request a failed connect.
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--gap", type=float, default=0.0, help="seconds between submissions (0: a burst)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fire").add_argument("sizes", nargs="+")
    sub.add_parser("compare").add_argument("sizes", nargs="*")
    sub.add_parser("mode").add_argument("state", choices=["batched", "8k", "4k", "2k"])
    args = parser.parse_args()

    if args.cmd == "mode":
        console.print("mode:", set_mode(args.url, args.state if args.state == "batched" else f"serial {args.state}"))
    elif args.cmd == "fire":
        show("results", *fire(args.url, args.sizes, args.gap))
    else:
        scenarios = {"your prompts": args.sizes} if args.sizes else SCENARIOS
        summary = {name: compare(args.url, sizes, args.gap) for name, sizes in scenarios.items()}
        console.print()
        for name, lines in summary.items():
            console.print(f"[bold]{name}[/] (batched):")
            for line in lines:
                console.print(f"  {line}")


if __name__ == "__main__":
    main()
