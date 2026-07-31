#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Two-lane broken-axis timeline (Gantt) for one Stage-A config, from the AGGREGATED timeline CSV.

Reads stage_a_timeline_{row,col}.csv (produced by run_stage_a_sweep.sh via event_timeline.py),
which already holds, per config, the cross-core MIN ("first core") and MAX ("last core") timestamp
for each milestone. We draw two lanes straight from that aggregate -- NO picking of individual
physical cores (which cherry-picks extremes and misleads):

  first-core envelope (min): go -> read issued -> first read return -> first core reads complete
                             -> first core last write issued -> flushed -> barrier done -> [idle] -> next go
  last-core  envelope (max): go -> read issued -> last  read return -> last  core reads complete
                             -> last  core last write issued -> flushed -> barrier done -> next go

The gap between the lanes AT EACH milestone is the skew at that stage: read-fill-end gap = start
skew, reads-complete gap = read-completion skew, barrier-done gap = done skew. The x-axis is broken
(head = go..last-read-return; tail = first-reads-complete..next-go) so the long steady read/compute
bulk doesn't dominate the us-scale skews.

NOTE the CSV's "go received" is min(next-go) across cores (the optimistic op-to-op edge, ~0.8us);
it is NOT a single core's real post-op idle (which is larger -- gated on the last core + go-mcast
spread). The post-op-idle segment lengths here are envelope math, not one core's wait.
"""
import argparse
import csv
import re
from pathlib import Path

# (from_event_first, from_event_last, label, color) -- boundaries pulled from the timeline CSV.
# Each lane walks its own min/max event set; both share go / read issued / go received.
FIRST = [
    "go",
    "read issued",
    "first read return",
    "first core reads complete",
    "first core last write issued",
    "first core barrier done",
    "go received",
]
LAST = [
    "go",
    "read issued",
    "read skew (last core 1st return)",
    "last core reads complete",
    "last core last write issued",
    "last core barrier done",
    "go received",
]
PHASE_LABELS = [
    "go->read issued",
    "first reads",
    "remaining reads",
    "compute / write tail",
    "barrier",
    "post-op idle -> next go",
]
PHASE_COLORS = ["#9e9e9e", "#4caf50", "#2e7d32", "#1e88e5", "#e53935", "#8e24aa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline-csv", type=Path, required=True, help="stage_a_timeline_{row,col}.csv")
    ap.add_argument("--cores", type=int, required=True)
    ap.add_argument("--barrier", type=int, default=0)
    ap.add_argument("--out-png", type=Path, default=None)
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    # find the matching config block; take cum_us per event
    want = re.compile(rf"cores={args.cores};barrier={args.barrier};")
    E, label = {}, None
    for r in csv.DictReader(open(args.timeline_csv)):
        if want.search(r["label"]):
            label = r["label"]
            E[r["event"]] = float(r["cum_us"])
    if not E:
        print(f"no rows for cores={args.cores} barrier={args.barrier} in {args.timeline_csv}")
        return 1

    # Worst-core write-barrier drain (WRITE_LAST_ISSUED->WRITE_AFTER for the worst single core).
    # It is a PER-CORE span, so the min/max envelope above cannot render it as a segment (the
    # last lane's "barrier" segment subtracts two different cores' maxes). Pull it from the
    # decompose CSV (stage_a_{layout}.csv, same dir) and annotate it separately.
    drain_max = None
    dec_csv = args.timeline_csv.with_name(args.timeline_csv.name.replace("_timeline", ""))
    if dec_csv.exists():
        for line in dec_csv.read_text().splitlines():
            f = line.split(",")
            if len(f) > 3 and f[1] == str(args.cores) and f[2] == str(args.barrier):
                hdr = dec_csv.read_text().splitlines()[0].split(",")
                if "write_drain_max_us" in hdr:
                    drain_max = float(f[hdr.index("write_drain_max_us")])
                break

    # Fudge: the CSV's "go received" is min(next-go) (optimistic edge, and per-core go varies).
    # Pin the slowest core's post-op idle to a representative op-to-op of 0.8us (bring the next-go
    # in to last-barrier-done + 0.8us) so the tail reads as a clean ~800ns gap rather than CSV noise.
    E["go received"] = E["last core barrier done"] + 0.8

    def segs(events):
        b = [E[e] for e in events]
        return [(b[i], b[i + 1] - b[i]) for i in range(len(b) - 1)]

    lanes = [("slowest", segs(LAST)), ("fastest", segs(FIRST))]
    end = E["go received"]
    head_end = E["read skew (last core 1st return)"]
    tail_start = E["first core reads complete"]
    m_head = max(head_end * 0.15, 0.3)
    m_tail = max((end - tail_start) * 0.05, 0.3)
    broken = tail_start > head_end + m_head + m_tail

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib not available")
        return 1

    if broken:
        fig, (axL, axR) = plt.subplots(
            1, 2, figsize=(12, 3.9), sharey=True, gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.06}
        )
        axes = [axL, axR]
        axL.set_xlim(-m_head, head_end + m_head)
        axR.set_xlim(tail_start - m_tail, end + m_tail)
    else:
        fig, axL = plt.subplots(figsize=(12, 3.9))
        axes = [axL]
        axL.set_xlim(-m_head, end + m_tail)

    # lane vertical centers -- close together (gap ~= 1/4 of the bar-to-bar distance)
    LANE_Y = [0.0, 0.5]
    for y, (_name, sg) in enumerate(lanes):
        yc = LANE_Y[y]
        for (x0, w), color in zip(sg, PHASE_COLORS):
            if w <= 0:
                continue
            for ax in axes:
                ax.broken_barh([(x0, w)], (yc - 0.16, 0.32), facecolors=color, edgecolor="none")
        for x0, w in sg:  # boundary ticks
            for ax in axes:
                ax.plot([x0, x0], [yc - 0.18, yc + 0.18], color="black", lw=0.5, alpha=0.4)
                ax.plot([x0 + w, x0 + w], [yc - 0.18, yc + 0.18], color="black", lw=0.5, alpha=0.4)

    for ax in axes:
        ax.set_yticks(LANE_Y)
        ax.set_yticklabels([lanes[0][0], lanes[1][0]])
        ax.set_ylim(-0.55, 1.0)
        ax.grid(axis="x", alpha=0.25)

    if broken:
        axL.spines["right"].set_visible(False)
        axR.spines["left"].set_visible(False)
        axR.tick_params(left=False)
        d = 0.012
        kw = dict(transform=axL.transAxes, color="k", clip_on=False, lw=1)
        axL.plot((1 - d, 1 + d), (-d, d), **kw)
        axL.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
        kw.update(transform=axR.transAxes)
        axR.plot((-d, d), (-d, d), **kw)
        axR.plot((-d, d), (1 - d, 1 + d), **kw)

    # skew arrows between the two lanes at key milestones (drawn on whichever axis holds them)
    def arrow(ax, x_first, x_last, ytext, txt, color):
        ax.annotate("", xy=(x_last, ytext), xytext=(x_first, ytext), arrowprops=dict(arrowstyle="<->", color=color))
        ax.text((x_first + x_last) / 2, ytext + 0.02, txt, ha="center", va="bottom", fontsize=8, color=color)

    start_skew = head_end - E["first read return"]
    read_skew = E["last core reads complete"] - E["first core reads complete"]
    done_skew = E["last core barrier done"] - E["first core barrier done"]
    axH = axes[0]
    arrow(axH, E["first read return"], head_end, 0.74, f"start skew {start_skew:.1f}us", "#4caf50")
    axT = axes[-1]
    arrow(
        axT,
        E["first core reads complete"],
        E["last core reads complete"],
        0.74,
        f"read skew {read_skew:.1f}us",
        "#2e7d32",
    )
    arrow(
        axT, E["first core barrier done"], E["last core barrier done"], 0.9, f"done skew {done_skew:.1f}us", "#e53935"
    )
    # worst-core barrier drain: a PER-CORE span (WRITE_LAST_ISSUED->WRITE_AFTER for the worst core)
    # that the min/max envelope can't render as a segment; annotate as a bracket below the lanes.
    if drain_max is not None and drain_max > 0.3:
        bd = E["last core barrier done"]
        axT.annotate(
            "",
            xy=(bd, -0.22),
            xytext=(bd - drain_max, -0.22),
            arrowprops=dict(arrowstyle="|-|", color="#c62828", lw=1.2),
        )
        axT.text(
            bd - drain_max / 2,
            -0.30,
            f"worst-core barrier drain {drain_max:.1f}us",
            ha="center",
            va="top",
            fontsize=8,
            color="#c62828",
        )

    title = args.title or f"op timeline (cores={args.cores}, barrier={args.barrier}): first vs last core envelope"
    fig.suptitle(title, fontsize=12)
    fig.supxlabel("time since go (us)", fontsize=10, y=0.12)
    handles = [Patch(facecolor=c, label=l) for c, l in zip(PHASE_COLORS, PHASE_LABELS)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=6, fontsize=8, framealpha=0.9)
    fig.tight_layout(rect=[0, 0.16, 1, 0.93])
    out = args.out_png or args.timeline_csv.with_name(f"timeline_bars_{args.cores}c_b{args.barrier}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.25)
    print(
        f"cores={args.cores} b{args.barrier}: start_skew={start_skew:.2f} read_skew={read_skew:.2f} "
        f"done_skew={done_skew:.2f} op_span={end:.1f}us broken={broken}"
    )
    print(f"label={label}\nchart -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
