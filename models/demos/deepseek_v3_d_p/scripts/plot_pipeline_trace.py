#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Render a pipeline-prefill run's behaviour as a Gantt chart from its log.

x = wall-clock time (s, relative to the first chunk start), y = rank. Each chunk is a bar on its
rank's row, coloured by chunk index, so the same chunk is the same colour on every rank — you watch
a chunk travel down the ranks (the pipeline diagonal). Handoff arrows trace chunk c from rank r to
rank r+1.

Input is the single combined run log (all ranks, MPI --tag-output interleaved). Parses the per-chunk
CHUNK_START epoch and the post-loop E2E_CLOCK; a chunk's bar runs start(r,c) -> start(r,c+1) (the
last chunk uses last_compute_end), i.e. compute + any idle wait for the next chunk.

    python -m models.demos.deepseek_v3_d_p.scripts.plot_pipeline_trace <run.log> [-o out.png]

Or, for gapless data under per-chunk sync, read the runner's per-rank timing CSVs directly instead of
the fd-race-prone combined log:

    python -m models.demos.deepseek_v3_d_p.scripts.plot_pipeline_trace --timing-dir <PREFILL_TIMING_DIR> [-o out.png]
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, no display
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Lines are MPI-tagged ([1,R]<stderr>:) and the message also carries [pp rank R]; key off the latter
# so the parser is independent of the MPI tag.
_CHUNK_START = re.compile(r"\[pp rank (\d+)\] CHUNK_START c=(\d+) compute_start=([\d.]+)")
_CHUNK_COMPUTE = re.compile(r"\[pp rank (\d+)\] CHUNK_COMPUTE c=(\d+) compute_ms=([\d.]+)")
_E2E = re.compile(r"\[pp rank (\d+)\] E2E_CLOCK first_compute_start=([\d.]+) last_compute_end=([\d.]+)")


def parse(path):
    """-> (starts[rank][chunk] = epoch, last_end[rank] = epoch, measured[rank][chunk] = compute seconds).

    measured is populated only when the runner ran with PREFILL_SYNC_PER_CHUNK (CHUNK_COMPUTE lines) —
    an exact per-chunk device compute, so the plot needs no downstream-start proxy or last-rank estimate."""
    starts = defaultdict(dict)
    measured = defaultdict(dict)
    last_end = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = _CHUNK_START.search(line)
            if m:
                starts[int(m.group(1))][int(m.group(2))] = float(m.group(3))
                continue
            m = _CHUNK_COMPUTE.search(line)
            if m:
                measured[int(m.group(1))][int(m.group(2))] = float(m.group(3)) / 1000.0
                continue
            m = _E2E.search(line)
            if m:
                last_end[int(m.group(1))] = float(m.group(3))
    if not starts:
        raise SystemExit(f"no CHUNK_START lines found in {path} (the pipeline loop always logs CHUNK_START)")
    return starts, last_end, measured


def parse_timing_dir(d):
    """-> the same (starts, last_end, measured) as parse(), from the per-rank timing CSVs the runner
    appends under PREFILL_TIMING_DIR (rank,c,compute_start,compute_ms). These bypass the shared-stdout
    fd race that splices C++ lines into the CHUNK_* log records, so this source is gapless."""
    starts = defaultdict(dict)
    measured = defaultdict(dict)
    last_end = {}
    for path in sorted(glob.glob(os.path.join(d, "rank*.csv"))):
        with open(path, errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 4:
                    continue
                try:
                    rank, c, cs, ms = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    continue
                starts[rank][c] = cs
                measured[rank][c] = ms / 1000.0
                last_end[rank] = max(last_end.get(rank, 0.0), cs + ms / 1000.0)
    if not starts:
        raise SystemExit(f"no rank*.csv timing rows found under {d}")
    return starts, last_end, measured


def trim_to_last(starts, last_end, measured, n):
    """Keep only the last n chunk indices, dropping the producer's warmup chunks so the chart shows the
    same real-request window the summarizer reports. last_end is rebuilt from the kept ends so the final
    bar isn't clipped."""
    all_c = sorted({c for r in starts for c in starts[r]})
    if n is None or n >= len(all_c):
        return starts, last_end, measured
    keep = set(all_c[-n:])
    starts = {r: d2 for r, d in starts.items() if (d2 := {c: v for c, v in d.items() if c in keep})}
    measured = {r: {c: v for c, v in measured.get(r, {}).items() if c in keep} for r in starts}
    last_end = {r: max(starts[r][c] + measured.get(r, {}).get(c, 0.0) for c in starts[r]) for r in starts}
    return starts, last_end, measured


def slot_end(starts, last_end, rank, chunk, ordered):
    """End of (rank, chunk)'s time SLOT (compute + idle): the next chunk's start, else last_compute_end."""
    i = ordered.index(chunk)
    if i + 1 < len(ordered):
        return starts[rank][ordered[i + 1]]
    return last_end.get(rank, starts[rank][chunk])


def median(xs):
    s = sorted(xs)
    n = len(s)
    return 0.0 if n == 0 else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", help="combined pipeline-prefill run log (all ranks)")
    ap.add_argument(
        "--timing-dir",
        help="dir of per-rank timing CSVs (PREFILL_TIMING_DIR); preferred over the log — gapless",
    )
    ap.add_argument("-o", "--out", default="pipeline_trace.png")
    ap.add_argument("--no-arrows", action="store_true", help="omit the rank->rank handoff arrows")
    ap.add_argument(
        "--real-chunks",
        type=int,
        default=None,
        help="plot only the last N chunks (drops producer warmup, matching the summarizer's window)",
    )
    args = ap.parse_args()

    if args.timing_dir:
        starts, last_end, measured = parse_timing_dir(args.timing_dir)
        source_label = os.path.basename(os.path.normpath(args.timing_dir))
    elif args.log:
        starts, last_end, measured = parse(args.log)
        source_label = args.log.split("/")[-1]
    else:
        ap.error("provide a log path or --timing-dir")

    starts, last_end, measured = trim_to_last(starts, last_end, measured, args.real_chunks)
    ranks = sorted(starts)
    origin = min(s for r in ranks for s in starts[r].values())  # t=0 at the first chunk start

    all_chunks = sorted({c for r in ranks for c in starts[r]})
    cpos = {c: i for i, c in enumerate(all_chunks)}
    n_chunks = len(all_chunks)
    cmap = plt.get_cmap("turbo", max(n_chunks, 1))

    # Per-chunk compute duration, proxied by the downstream rank's start (it unblocks ~immediately when
    # this rank's output arrives; transport is ~ms). The last rank has no downstream proxy, so its
    # compute is estimated as the median of all proxied durations and its bars are hatched lighter.
    def compute_dur(rank, c):
        if rank + 1 in starts and c in starts[rank + 1]:
            return max(starts[rank + 1][c] - starts[rank][c], 0.0)
        return None  # last rank / no proxy

    proxied = [compute_dur(r, c) for r in ranks[:-1] for c in starts[r] if compute_dur(r, c) is not None]
    est_compute = median(proxied)  # fallback for the last rank

    fig, ax = plt.subplots(figsize=(14, 2 + 1.1 * len(ranks)))
    bar_h = 1.0  # full-height rows: ranks stack flush, no vertical gaps

    for rank in ranks:
        ordered = sorted(starts[rank])
        for c in ordered:
            s = starts[rank][c]
            slot_e = slot_end(starts, last_end, rank, c, ordered)
            # Exact measured compute (PREFILL_SYNC_PER_CHUNK) wins over the downstream-start proxy.
            dur = measured.get(rank, {}).get(c)
            estimated = dur is None
            if estimated:
                dur = compute_dur(rank, c)
                estimated = dur is None
            if estimated:
                # last rank: clamp the upstream-median estimate to the slot. With no proxy data at all
                # (single rank), there is no pipeline overlap, so the whole slot IS the compute.
                slot = slot_e - s
                if est_compute > 0:
                    # slot<=0 means the final chunk has no end marker (E2E_CLOCK didn't flush, so
                    # last_end is empty and slot_e degenerates to s); use the estimate so the last
                    # rank's last bar isn't drawn zero-width.
                    dur = min(est_compute, slot) if slot > 0 else est_compute
                else:
                    own = [
                        b - a
                        for a, b in zip([starts[rank][x] for x in ordered], [starts[rank][x] for x in ordered[1:]])
                    ]
                    dur = slot if slot > 0 else median(own)
            comp_end = s + dur
            color = cmap(cpos[c])
            # Compute block (solid). The idle time (comp_end -> next chunk start) is left UNDRAWN, so
            # the white background shows through as the pipeline bubble.
            ax.broken_barh(
                [(s - origin, max(dur, 1e-3))],
                (rank - bar_h / 2, bar_h),
                facecolors=color,
                edgecolors="black",
                linewidths=0.4,
                hatch="//" if estimated else None,
            )
            if n_chunks <= 16:
                ax.text(
                    (s - origin) + max(dur, 1e-3) / 2,
                    rank,
                    str(c),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                )

    # Handoff arrows: chunk c leaves rank r and starts on rank r+1 at start(r+1, c). Only meaningful
    # when rows have a vertical gap; with flush (full-height) bars the same-colour diagonal IS the
    # handoff, so arrows are skipped.
    if not args.no_arrows and bar_h < 0.95:
        for rank in ranks[:-1]:
            nxt = rank + 1
            if nxt not in starts:
                continue
            for c in sorted(set(starts[rank]) & set(starts[nxt])):
                x0 = starts[rank][c] - origin
                x1 = starts[nxt][c] - origin
                ax.add_patch(
                    FancyArrowPatch(
                        (x0, rank + bar_h / 2),
                        (x1, nxt - bar_h / 2),
                        arrowstyle="-|>",
                        mutation_scale=7,
                        color=cmap(cpos[c]),
                        alpha=0.5,
                        linewidth=0.7,
                        shrinkA=0,
                        shrinkB=0,
                    )
                )

    from matplotlib.patches import Patch

    # With CHUNK_COMPUTE every bar is exact, so the proxy/estimate legend entries would misdescribe it.
    any_measured = any(measured.get(r) for r in ranks)
    handles = [
        Patch(
            facecolor="grey",
            edgecolor="black",
            label="compute (measured, per-chunk sync)" if any_measured else "compute (downstream-start proxy)",
        ),
        Patch(facecolor="white", edgecolor="grey", label="idle / waiting (white gap)"),
    ]
    if not any_measured:
        handles.append(
            Patch(facecolor="grey", hatch="//", edgecolor="black", label="last rank: compute estimated (median)")
        )
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_yticks(ranks)
    ax.set_yticklabels([f"rank {r}" for r in ranks])
    # rank 0 at the bottom; the pipeline flows upward (default y increases upward).
    ax.set_xlabel("time since first chunk start (s)")
    ax.set_title(f"pipeline-prefill trace — {len(ranks)} ranks, {n_chunks} chunks  ({source_label})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  ({len(ranks)} ranks, {n_chunks} chunks)")


if __name__ == "__main__":
    main()
