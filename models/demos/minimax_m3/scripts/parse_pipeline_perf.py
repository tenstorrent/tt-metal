# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Steady-state throughput and per-request latency from pipeline-prefill runner logs.

The shared runner (models/demos/common/prefill/runners/prefill_runner.py) logs one line per chunk per rank:
    [pp rank {r}] CHUNK_START c={c} compute_start={epoch:.6f} slot={s} [{actual_start},{actual_end}) ...
`c` counts the chunks a rank has processed, so one chunk carries the same `c` on every rank.

Throughput: a pipeline emits one chunk per period of its slowest stage. A rank's period is the median gap between
consecutive compute_start values with the first FILL_DROP and the last DRAIN_DROP gaps removed; the bottleneck rank
has the largest period; tok/s = tokens per chunk / period. The ceiling is tokens per chunk over the fastest gap any
rank showed: a steady rate far below it means the pipeline is starved (producer or socket bound), not device bound.

Latency (TTFT): prefill is KV-only, so a request runs from its first chunk's compute_start on the first rank to its
last chunk's compute_start on the last rank. A slot is reused across requests, so a chunk with actual_start 0 opens
a new one; requests are paired across ranks by the `c` of that opening chunk.

summarize_ci_run.py (models/demos/common/prefill/runners/ci) reads the same line for CI and rates one request's
completion cadence. This script takes the median over a multi-request round-robin run, which is what a steady-state
A/B needs.

    python models/demos/minimax_m3/scripts/parse_pipeline_perf.py <runner.log> [...]
"""

import argparse
import math
import re
import statistics
import sys
from collections import Counter

_CHUNK_START = re.compile(
    r"\[pp rank (?P<rank>\d+)\] CHUNK_START c=(?P<c>\d+) compute_start=(?P<t>[\d.]+) "
    r"slot=(?P<slot>\d+) \[(?P<actual_start>\d+),(?P<actual_end>\d+)\)"
)

# Chunk 0 compiles and chunk 1 still runs without upstream back-pressure, so neither gap is steady state; the final
# gap is the drain. Enough for the 2- and 4-stage runs.
FILL_DROP = 2
DRAIN_DROP = 1


def parse_events(paths):
    """[{rank, c, t, slot, actual_start, actual_end}] from one or more runner logs, in file order."""
    events = []
    for path in paths:
        with open(path, errors="ignore") as fh:
            for line in fh:
                m = _CHUNK_START.search(line)
                if m:
                    events.append({k: (float(v) if k == "t" else int(v)) for k, v in m.groupdict().items()})
    return events


def tokens_per_chunk(events):
    """The most common [actual_start, actual_end) span: a request's final chunk may be shorter."""
    return Counter(e["actual_end"] - e["actual_start"] for e in events).most_common(1)[0][0]


def steady_state_throughput(events, chunk_tokens):
    """Per-rank steady-state period and the bottleneck stage's throughput. Returns (summary, per_rank)."""
    per_rank = {}
    for rank in sorted({e["rank"] for e in events}):
        ts = sorted(e["t"] for e in events if e["rank"] == rank)
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        body = gaps[FILL_DROP : len(gaps) - DRAIN_DROP] if len(gaps) > FILL_DROP + DRAIN_DROP else gaps
        if body:
            per_rank[rank] = {
                "chunks": len(ts),
                "median_gap_ms": statistics.median(body) * 1000.0,
                "min_gap_ms": min(body) * 1000.0,
            }
    if not per_rank:
        return {"bottleneck_rank": None, "steady_period_ms": 0.0, "steady_tok_s": 0.0, "ceiling_tok_s": 0.0}, per_rank
    bottleneck = max(per_rank, key=lambda r: per_rank[r]["median_gap_ms"])
    steady_s = per_rank[bottleneck]["median_gap_ms"] / 1000.0
    fastest_s = min(v["min_gap_ms"] for v in per_rank.values()) / 1000.0
    return (
        {
            "bottleneck_rank": bottleneck,
            "steady_period_ms": steady_s * 1000.0,
            "steady_tok_s": chunk_tokens / steady_s,
            "ceiling_tok_s": chunk_tokens / fastest_s,
        },
        per_rank,
    )


def _requests(rank_events):
    """One rank's chunks grouped into requests, keyed by the opening chunk's c: within a slot, a chunk with
    actual_start 0 opens a new request."""
    requests = {}
    for slot in {e["slot"] for e in rank_events}:
        opening = None
        for e in sorted((e for e in rank_events if e["slot"] == slot), key=lambda e: e["t"]):
            if e["actual_start"] == 0 or opening is None:
                opening = e["c"]
                requests[opening] = []
            requests[opening].append(e)
    return requests


def per_request_latency(events):
    """Per-request prefill latency: the last rank's final-chunk compute_start minus the first rank's opening-chunk
    compute_start, paired by the opening chunk's c. Returns (p50_s, p90_s, n_requests)."""
    ranks = sorted({e["rank"] for e in events})
    first = _requests([e for e in events if e["rank"] == ranks[0]])
    last = _requests([e for e in events if e["rank"] == ranks[-1]])
    lats = sorted(last[c][-1]["t"] - first[c][0]["t"] for c in first if c in last)
    if not lats:
        return 0.0, 0.0, 0
    p90 = lats[max(0, math.ceil(0.9 * len(lats)) - 1)]  # nearest-rank percentile
    return statistics.median(lats), p90, len(lats)


def summarize(paths, chunk_tokens=None):
    events = parse_events(paths)
    if not events:
        raise ValueError(f"no CHUNK_START lines in {paths}")
    chunk_tokens = chunk_tokens or tokens_per_chunk(events)
    throughput, per_rank = steady_state_throughput(events, chunk_tokens)
    p50, p90, n_requests = per_request_latency(events)
    return {
        "n_events": len(events),
        "chunk_tokens": chunk_tokens,
        "per_rank": per_rank,
        "throughput": throughput,
        "latency_p50_s": p50,
        "latency_p90_s": p90,
        "n_requests": n_requests,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="runner log file(s)")
    ap.add_argument("--chunk-size", type=int, default=None, help="tokens per chunk (default: read from the log)")
    args = ap.parse_args(argv)
    try:
        s = summarize(args.logs, args.chunk_size)
    except ValueError as err:
        sys.exit(f"parse_pipeline_perf: {err}")
    print(f"events={s['n_events']} requests={s['n_requests']} chunk_tokens={s['chunk_tokens']}")
    for rank, r in sorted(s["per_rank"].items()):
        print(
            f"  rank {rank}: {r['chunks']} chunks | steady gap {r['median_gap_ms']:.1f} ms | min {r['min_gap_ms']:.1f} ms"
        )
    t = s["throughput"]
    print(
        f"BOTTLENECK rank {t['bottleneck_rank']} | steady {t['steady_period_ms']:.1f} ms/chunk "
        f"=> {t['steady_tok_s']:.0f} tok/s (steady) | {t['ceiling_tok_s']:.0f} tok/s (min-gap ceiling)"
    )
    print(f"TTFT (prefill KV-ready) p50={s['latency_p50_s'] * 1000:.0f} ms p90={s['latency_p90_s'] * 1000:.0f} ms")


if __name__ == "__main__":
    main()
