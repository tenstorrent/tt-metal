# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-chunk prefill timings from a prefill_runner log, and the KV-depth ramp they show.

Usage:  analyze_prefill_per_chunk.py <runner.log> [--rank N] [--request N] [--csv out.csv]

Cost per chunk grows with KV depth, which a single median throughput figure hides. A chunk's cost
is the interval to the next CHUNK_START on the same rank, so it includes host work between issues
and there is no interval for the last chunk; CHUNK_END closes that gap when the runner logs it.

For P>1 the period and the traversal are different quantities and must not be conflated:

    period    = last_rank_start(c+1) - last_rank_start(c)    [throughput]
    traversal = last_rank_finish(c)  - rank0_start(c)        [latency: time in the pipeline]

Without CHUNK_END the finish is approximated by the last rank's next start, exact only while that
rank stays saturated.

See also analyze_prefill_kv_ramp.py, which attributes the growth to specific ops.
"""

import argparse
import re
import statistics
from typing import NamedTuple

CHUNK_START = re.compile(
    r"\[pp rank (\d+)\] CHUNK_START c=(\d+) compute_start=([0-9.]+) slot=(?:-?\d+) \[(\d+),(\d+)\)"
)
CHUNK_END = re.compile(r"\[pp rank (\d+)\] CHUNK_END c=(\d+) compute_end=([0-9.]+)")


class Chunk(NamedTuple):
    tok_start: int
    tok_end: int
    ms: float  # the period on a pipeline, the chunk's duration on single rank
    traversal_ms: float  # equals ms on single rank, so both topologies share one schema

    @property
    def tok_s(self) -> float:
        return (self.tok_end - self.tok_start) / (self.ms / 1000.0)


def split_requests(rows):
    """[[ (c, t, tok_start, tok_end) ]] -- a new request restarts at tok_start == 0."""
    out, cur = [], []
    for row in rows:
        if row[2] == 0 and cur:
            out.append(cur)
            cur = []
        cur.append(row)
    if cur:
        out.append(cur)
    return out


def parse(path):
    """-> starts{rank: [(c, t, tok_start, tok_end)]}, ends{rank: {c: t}}.

    `c` is a GLOBAL per-process counter, not per-request: a 2-request 51-chunk run logs c=0..101.
    CHUNK_END is keyed the same way, so pairing the two by within-request position differences
    request 2's start against request 1's end and yields negative durations.
    """
    txt = open(path, errors="replace").read()
    starts, ends = {}, {}
    for rk, c, t, st, en in CHUNK_START.findall(txt):
        starts.setdefault(int(rk), []).append((int(c), float(t), int(st), int(en)))
    for rk in starts:
        starts[rk].sort()
    for rk, c, t in CHUNK_END.findall(txt):
        ends.setdefault(int(rk), {})[int(c)] = float(t)
    return starts, ends


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="a prefill_runner runner.log")
    ap.add_argument("--rank", type=int, default=None, help="default: the last rank, which retires chunks")
    ap.add_argument("--request", type=int, default=None, help="1-based; default: the last (warm) request")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    starts, ends = parse(a.log)
    if not starts:
        raise SystemExit(f"no CHUNK_START records in {a.log} -- is this a completed runner log?")

    last_rank, first_rank = max(starts), min(starts)
    rank = a.rank if a.rank is not None else last_rank
    if rank not in starts:
        raise SystemExit(f"rank {rank} not in log; found {sorted(starts)}")

    reqs = split_requests(starts[rank])
    idx = (a.request - 1) if a.request is not None else len(reqs) - 1
    if not 0 <= idx < len(reqs):
        raise SystemExit(f"request {a.request} not in log; found {len(reqs)}")
    req = reqs[idx]

    # Rank 0's copy of the same request, for the traversal column.
    r0 = None
    if rank != first_rank:
        r0_reqs = split_requests(starts[first_rank])
        if idx < len(r0_reqs) and len(r0_reqs[idx]) == len(req):
            r0 = r0_reqs[idx]

    # Every chunk of THIS request needs an end stamp keyed by its global c. A run can log one extra
    # CHUNK_END at teardown, so counting lines is not the test.
    end_map = ends.get(rank, {})
    have_ends = all(c in end_map for c, _t, _st, _en in req)

    per = []
    for i, (c, t, st, en) in enumerate(req):
        if have_ends:
            fin = end_map[c]
        elif i + 1 < len(req):
            fin = req[i + 1][1]
        else:
            continue
        ms = (fin - t) * 1000.0
        if ms <= 0:
            raise SystemExit(f"non-positive duration {ms:.1f}ms for chunk c={c} -- stamps are mispaired")
        per.append(Chunk(st, en, ms, (fin - r0[i][1]) * 1000.0 if r0 else ms))

    if not per:
        raise SystemExit("only one chunk and no CHUNK_END stamps -- nothing to report")

    n = len(req)
    print(f"log    : {a.log}")
    print(f"rank   : {rank} (of 0..{last_rank})" + ("  <- last rank, retires chunks" if rank == last_rank else ""))
    print(f"request: {idx + 1} of {len(reqs)}" + ("  (warm)" if idx else "  (COLD -- includes trace capture)"))
    print(
        "source : "
        + ("CHUNK_END device-completion stamps" if have_ends else "CHUNK_START deltas (last chunk unavailable)")
    )
    if r0:
        print("columns: PERIOD = how often a chunk retires (throughput).")
        print("         TRAVERSAL = how long that chunk spent crossing all stages (latency).")
        print("         per-chunk tok/s is omitted: on a pipeline it is a rate, not this chunk's speed.")
    print()
    for i, ch in enumerate(per):
        head = f"[per_chunk] chunk {i + 1}/{n} [{ch.tok_start}, {ch.tok_end})"
        if r0:
            print(f"{head}  period={ch.ms:7.1f}ms  traversal={ch.traversal_ms:8.1f}ms")
        else:
            print(f"{head} {ch.ms:8.1f}ms ({ch.tok_s:,.0f} tok/s)")

    if r0:
        travs = [ch.traversal_ms for ch in per]
        print(
            f"\n[per_chunk] PIPELINE FILL (rank {first_rank} starts chunk 1 -> rank {rank} starts it): "
            f"{(req[0][1] - r0[0][1]) * 1000.0:.1f} ms"
        )
        print(
            f"[per_chunk] traversal: first={travs[0]:.1f}ms last={travs[-1]:.1f}ms "
            f"min={min(travs):.1f}ms max={max(travs):.1f}ms  <- what one chunk costs end to end"
        )

    ms_list = [ch.ms for ch in per]
    tok = per[-1].tok_end - per[0].tok_start
    total_s = sum(ms_list) / 1000.0
    print(
        f"\n[per_chunk] TOTAL {tok:,} tokens over {len(per)} timed chunks in {total_s:.1f}s "
        f"({tok / total_s:,.0f} tok/s) | mean={statistics.mean(ms_list):.1f}ms "
        f"min={min(ms_list):.1f}ms max={max(ms_list):.1f}ms"
    )
    line = f"[per_chunk] median={statistics.median(ms_list):.1f}ms"
    if len(ms_list) > 8:
        line += f" | median after warmup 8={statistics.median(ms_list[8:]):.1f}ms"
    print(line)
    if len(ms_list) > 8 and len(reqs) > 1:
        print(
            f"[per_chunk] note: that is one request's deep-KV tail; analyze_prefill_throughput.py flattens all "
            f"{len(reqs)} requests, so the two medians differ by construction (~7% on a 51-chunk pair)."
        )
    elif len(ms_list) <= 8:
        print(f"[per_chunk] note: only {len(ms_list)} chunks, too few to discard the usual 8 -- this is all ramp")
    print(
        f"[per_chunk] KV-depth cost: first={ms_list[0]:.1f}ms -> last={ms_list[-1]:.1f}ms = "
        f"{ms_list[-1] / ms_list[0]:.2f}x over {len(ms_list) - 1} extra chunks of history"
    )
    if not have_ends:
        print(
            f"[per_chunk] note: chunk {n}/{n} has no successor to difference against and is omitted. "
            f"A runner that logs CHUNK_END stamps gives real per-chunk device times."
        )

    if a.csv:
        with open(a.csv, "w") as fh:
            fh.write("chunk,tok_start,tok_end,period_ms,traversal_ms,tok_s\n")
            for i, ch in enumerate(per):
                fh.write(f"{i},{ch.tok_start},{ch.tok_end},{ch.ms:.3f},{ch.traversal_ms:.3f},{ch.tok_s:.1f}\n")
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
