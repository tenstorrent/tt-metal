#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Render a pipeline-prefill run's behaviour as a Gantt chart.

x = wall-clock time (s, relative to the first chunk), y = rank. Each chunk is a bar on its rank's row,
coloured by chunk index, so the same chunk is the same colour on every rank — you watch a chunk travel
down the ranks (the pipeline diagonal). White gaps are idle time.

From a run's rank-log directory (preferred): the producer's CHUNK_RETIRE lines carry every layer ack of
a chunk on the producer clock, and each runner rank logs the layers it owns. A rank's bar runs from the
upstream rank's last ack for that chunk (or its own previous chunk's last ack, whichever is later) to
its own last ack. Rank 0 has no upstream, so its bar starts one mean ack gap before its first ack.

    python -m models.demos.deepseek_v3_d_p.scripts.plot_pipeline_trace --ranklogs <dir> [-o out.png]

From a single combined run log (all ranks): CHUNK_START per rank plus the post-loop E2E_CLOCK. A chunk's
compute is proxied by the downstream rank's start; the last rank's is the median of those (hatched).

    python -m models.demos.deepseek_v3_d_p.scripts.plot_pipeline_trace <run.log> [-o out.png]
"""

import argparse
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, no display
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Lines are MPI-tagged ([1,R]<stderr>:) and the message also carries [pp rank R]; key off the latter
# so the parser is independent of the MPI tag.
_CHUNK_START = re.compile(r"\[pp rank (\d+)\] CHUNK_START c=(\d+) compute_start=([\d.]+)")
_E2E = re.compile(r"\[pp rank (\d+)\] E2E_CLOCK first_compute_start=([\d.]+) last_compute_end=([\d.]+)")
_RANK_LAYERS = re.compile(r"\[pp rank (\d+)/\d+\] .*? layers=\[(\d+), (\d+)\)")
_CHUNK_RETIRE = re.compile(
    r"\[producer\] CHUNK_RETIRE slot=(\d+) start=(\d+) end=(\d+) first=([\d.]+) retire=[\d.]+ warmup=([01])"
    r" acks=\d+ layers=(\S+)"
)


def median(xs):
    s = sorted(xs)
    n = len(s)
    return 0.0 if n == 0 else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def bars_from_ranklogs(root, real_chunks):
    """-> {rank: [(chunk_pos, start, dur, estimated)]} from the producer's layer acks and the runners' layer ranges."""
    ranges = {}
    retires = set()
    for dirpath, _, names in os.walk(root):
        for name in names:
            with open(os.path.join(dirpath, name), errors="ignore") as f:
                for line in f:
                    m = _RANK_LAYERS.search(line)
                    if m:
                        ranges[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
                        continue
                    m = _CHUNK_RETIRE.search(line)
                    if m and m.group(5) == "0":
                        retires.add((float(m.group(4)), m.group(6)))
    if len(ranges) < 2:
        raise SystemExit(f"need the layer ranges of 2+ runner ranks under {root}, found {sorted(ranges)}")
    if not retires:
        raise SystemExit(f"no measured CHUNK_RETIRE lines with layers= under {root}")
    last_rank = max(ranges)

    def rank_of(layer):
        for r, (lo, hi) in ranges.items():
            if lo <= layer < hi:
                return r
        return last_rank

    chunks = []
    for first, spans in sorted(retires):
        first_ack, last_ack, n_acks = {}, {}, defaultdict(int)
        for span in spans.split(","):
            layers, ms = span.split("@")
            t = first + float(ms) / 1000.0
            r = rank_of(int(layers.split("-")[0]))
            first_ack[r] = min(first_ack.get(r, t), t)
            last_ack[r] = max(last_ack.get(r, t), t)
            n_acks[r] += 1
        for r in first_ack:
            if n_acks[r] > 1:
                first_ack[r] -= (last_ack[r] - first_ack[r]) / (n_acks[r] - 1)
        chunks.append((first_ack, last_ack))
    if real_chunks is not None:
        chunks = chunks[-real_chunks:]

    bars = defaultdict(list)
    prev_end = {}
    for pos, (first_ack, last_ack) in enumerate(chunks):
        for r in sorted(last_ack):
            start = last_ack.get(r - 1, first_ack[r])
            start = min(max(start, prev_end.get(r, start)), last_ack[r])
            bars[r].append((pos, start, last_ack[r] - start, False))
            prev_end[r] = last_ack[r]
    return bars, len(chunks)


def bars_from_log(path, real_chunks):
    """-> {rank: [(chunk_pos, start, dur, estimated)]} from CHUNK_START with the downstream-start proxy."""
    starts = defaultdict(dict)
    last_end = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = _CHUNK_START.search(line)
            if m:
                starts[int(m.group(1))][int(m.group(2))] = float(m.group(3))
                continue
            m = _E2E.search(line)
            if m:
                last_end[int(m.group(1))] = float(m.group(3))
    if not starts:
        raise SystemExit(f"no CHUNK_START lines found in {path} (the pipeline loop always logs CHUNK_START)")

    all_c = sorted({c for r in starts for c in starts[r]})
    if real_chunks is not None and real_chunks < len(all_c):
        keep = set(all_c[-real_chunks:])
        starts = {r: d2 for r, d in starts.items() if (d2 := {c: v for c, v in d.items() if c in keep})}
        all_c = sorted(keep)
    cpos = {c: i for i, c in enumerate(all_c)}
    ranks = sorted(starts)

    def compute_dur(rank, c):
        if rank + 1 in starts and c in starts[rank + 1]:
            return max(starts[rank + 1][c] - starts[rank][c], 0.0)
        return None

    est_compute = median([d for r in ranks[:-1] for c in starts[r] if (d := compute_dur(r, c)) is not None])

    bars = defaultdict(list)
    for rank in ranks:
        ordered = sorted(starts[rank])
        for i, c in enumerate(ordered):
            s = starts[rank][c]
            dur = compute_dur(rank, c)
            estimated = dur is None
            if estimated:
                slot_e = starts[rank][ordered[i + 1]] if i + 1 < len(ordered) else last_end.get(rank, s)
                slot = slot_e - s
                if est_compute > 0:
                    dur = min(est_compute, slot) if slot > 0 else est_compute
                else:
                    own = [starts[rank][b] - starts[rank][a] for a, b in zip(ordered, ordered[1:])]
                    dur = slot if slot > 0 else median(own)
            bars[rank].append((cpos[c], s, dur, estimated))
    return bars, len(all_c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", help="combined pipeline-prefill run log (all ranks)")
    ap.add_argument("--ranklogs", help="run dir holding the producer and runner rank logs; preferred over the log")
    ap.add_argument("-o", "--out", default="pipeline_trace.png")
    ap.add_argument(
        "--real-chunks",
        type=int,
        default=None,
        help="plot only the last N chunks (drops producer warmup, matching the summarizer's window)",
    )
    args = ap.parse_args()

    if args.ranklogs:
        bars, n_chunks = bars_from_ranklogs(args.ranklogs, args.real_chunks)
        source_label = os.path.basename(os.path.normpath(args.ranklogs))
        compute_label = "compute (layer acks, producer clock)"
    elif args.log:
        bars, n_chunks = bars_from_log(args.log, args.real_chunks)
        source_label = os.path.basename(args.log)
        compute_label = "compute (downstream-start proxy)"
    else:
        ap.error("provide a log path or --ranklogs")

    ranks = sorted(bars)
    origin = min(s for r in ranks for _, s, _, _ in bars[r])
    cmap = plt.get_cmap("turbo", max(n_chunks, 1))

    fig, ax = plt.subplots(figsize=(14, 2 + 1.1 * len(ranks)))
    for rank in ranks:
        for pos, s, dur, estimated in bars[rank]:
            width = max(dur, 1e-3)
            ax.broken_barh(
                [(s - origin, width)],
                (rank - 0.5, 1.0),
                facecolors=cmap(pos),
                edgecolors="black",
                linewidths=0.4,
                hatch="//" if estimated else None,
            )
            if n_chunks <= 16:
                ax.text(
                    (s - origin) + width / 2,
                    rank,
                    str(pos),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                )

    handles = [
        Patch(facecolor="grey", edgecolor="black", label=compute_label),
        Patch(facecolor="white", edgecolor="grey", label="idle / waiting (white gap)"),
    ]
    if any(e for r in ranks for *_, e in bars[r]):
        handles.append(
            Patch(facecolor="grey", hatch="//", edgecolor="black", label="last rank: compute estimated (median)")
        )
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_yticks(ranks)
    ax.set_yticklabels([f"rank {r}" for r in ranks])
    ax.set_xlabel("time since first chunk (s)")
    ax.set_title(f"pipeline-prefill trace — {len(ranks)} ranks, {n_chunks} chunks  ({source_label})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  ({len(ranks)} ranks, {n_chunks} chunks)")


if __name__ == "__main__":
    main()
