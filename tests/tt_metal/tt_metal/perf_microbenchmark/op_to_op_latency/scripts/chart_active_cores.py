#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Active-cores-over-time for one op, to visualize the straggler tail that stretches the envelope.

Each core is "active" (demanding DRAM bandwidth) from its first read to its write-barrier done. We count
how many of the N cores are active at each instant and plot that step curve. BW saturates at ~SAT cores,
so any interval where fewer than SAT cores are active runs the DRAM pipe under-utilized: the ramp-up fill
and -- the point of interest -- the ragged FINISH tail, where the last stragglers run mostly alone. That
sub-SAT tail is time the 8-core config doesn't pay (it finishes tight) and is why more cores are slower.

Input: raw device-profiler CSV(s). Pass one or two (e.g. 8c and 56c) to overlay-compare in stacked panels.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_op_to_op_profiler_csv import load_profiler_csv, parse_chip_freq_mhz  # noqa: E402
from chart_timeline_k2 import per_core_invocations  # noqa: E402


def intervals(csv):
    freq = parse_chip_freq_mhz(Path(csv).resolve())
    df = load_profiler_csv(Path(csv).resolve())
    inv = {k: v for k, v in per_core_invocations(df).items() if v}
    t0 = min(v[-1]["go"] for v in inv.values())
    out = []
    for v in inv.values():
        op = v[-1]  # last steady op
        start = op["read_bursts"][0][0] if op["read_bursts"] else op["go"]
        out.append(((start - t0) / freq, (op["wafter"] - t0) / freq))
    return out


def step_curve(ivals):
    ev = []
    for s, f in ivals:
        ev.append((s, 1))
        ev.append((f, -1))
    ev.sort()
    ts, cs, c = [0.0], [0], 0
    for t, d in ev:
        ts.append(t)
        cs.append(c)
        c += d
        ts.append(t)
        cs.append(c)
    return ts, cs


def sub_sat_tail(ivals, sat):
    """Time from the moment active-count LAST falls below `sat` (on the decay) to the final finish."""
    finishes = sorted(f for _, f in ivals)
    n = len(finishes)
    # active count drops below sat once (n - sat + 1) cores have finished... i.e. when only (sat-1) remain
    if n < sat:
        return finishes[0], finishes[-1]
    cross = finishes[n - sat + 1] if (n - sat + 1) < n else finishes[-1]  # (sat-1) cores remain after this
    return cross, finishes[-1]


def bw_over_time(ivals, bytes_per_core, nb=240):
    """Approximate delivered aggregate BW (GB/s) over time: spread each core's read+write bytes uniformly
    over its active span, sum per time bin. Shows the gradual roll-off as fast cores leave and only slow
    (NoC-distant) stragglers remain -- the real under-utilization, which begins ABOVE the sat-core line."""
    last = max(f for _, f in ivals)
    T = last * 1.02
    binw = T / nb
    bins = [0.0] * nb
    for s, f in ivals:
        span = max(f - s, 1e-6)
        for bi in range(nb):
            t0, t1 = bi * binw, (bi + 1) * binw
            ov = min(f, t1) - max(s, t0)
            if ov > 0:
                bins[bi] += bytes_per_core * (ov / span)  # bytes delivered in this bin by this core
    xs = [(bi + 0.5) * binw for bi in range(nb)]
    gbps = [b / binw * 1e-3 for b in bins]  # bytes/µs -> GB/s
    return xs, gbps


def panel(ax, csv, label, sat, ceiling, color):
    ivals = intervals(csv)
    n = len(ivals)
    pages_per_core = 3584 // n  # fixed-total work model
    bytes_per_core = 2 * pages_per_core * 2048  # read + write
    ts, cs = step_curve(ivals)
    ax.fill_between(ts, cs, step="post", color=color, alpha=0.20)
    ax.plot(ts, cs, drawstyle="steps-post", color=color, lw=1.3, label="active cores")
    ax.axhline(sat, ls="--", color="#b71c1c", lw=1.0)
    ax.text(0, sat + 0.5, f"≈{sat} cores saturate BW", ha="left", va="bottom", fontsize=7, color="#b71c1c")
    first_fin = min(f for _, f in ivals)
    last_fin = max(f for _, f in ivals)
    ax.axvline(first_fin, ls=":", color="#555", lw=0.8)
    ax.text(first_fin, n * 0.98, f"1st done {first_fin:.0f}µs", rotation=90, fontsize=7, va="top", ha="right")
    ax.axvline(last_fin, ls=":", color="#555", lw=0.8)
    ax.text(last_fin, n * 0.98, f"last {last_fin:.0f}µs", rotation=90, fontsize=7, va="top", ha="left")
    ax.set_ylabel("active cores", color=color)
    ax.set_ylim(0, n * 1.08)
    ax.grid(alpha=0.2)

    # delivered BW over time on twin axis -- the truthful under-utilization view
    ax2 = ax.twinx()
    xs, gbps = bw_over_time(ivals, bytes_per_core)
    ax2.plot(xs, gbps, color="#e65100", lw=2.0, label="delivered BW")
    ax2.axhline(ceiling, ls="--", color="#e65100", lw=1.0, alpha=0.7)
    ax2.text(0, ceiling + 4, f"~{ceiling} GB/s ceiling", ha="left", va="bottom", fontsize=7, color="#e65100")
    # shade where delivered BW has rolled off below 90% of ceiling AND we're past peak (the real tail)
    peak_i = max(range(len(gbps)), key=lambda i: gbps[i])
    roll = next((xs[i] for i in range(peak_i, len(xs)) if gbps[i] < 0.9 * ceiling), last_fin)
    ax2.axvspan(roll, last_fin, color="#e65100", alpha=0.10)
    ax2.annotate(
        f"BW roll-off tail\n{last_fin - roll:.0f}µs\n(slow stragglers can't\nfill freed BW)",
        xy=((roll + last_fin) / 2, ceiling * 0.55),
        ha="center",
        va="center",
        fontsize=7.5,
        color="#bf360c",
        fontweight="bold",
    )
    ax2.set_ylabel("delivered BW (GB/s)", color="#e65100")
    ax2.set_ylim(0, ceiling * 1.15)
    ax.set_title(
        f"{label}: {n} cores | envelope {last_fin:.0f}µs | finish spread {last_fin - first_fin:.0f}µs | "
        f"BW roll-off tail {last_fin - roll:.0f}µs",
        fontsize=10,
        loc="left",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="device CSV (repeat for stacked panels)")
    ap.add_argument("--label", action="append", required=True, help="label per --csv")
    ap.add_argument("--sat-cores", type=int, default=8, help="cores needed to saturate DRAM BW")
    ap.add_argument("--ceiling", type=float, default=206.0, help="aggregate bidirectional BW ceiling (GB/s)")
    ap.add_argument("--out-png", type=Path, required=True)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncsv = len(args.csv)
    fig, axes = plt.subplots(ncsv, 1, figsize=(12, 3.2 * ncsv), sharex=True)
    if ncsv == 1:
        axes = [axes]
    colors = ["#1e88e5", "#2e7d32", "#8e24aa"]
    for ax, csv, lab, col in zip(axes, args.csv, args.label, colors):
        panel(ax, csv, lab, args.sat_cores, args.ceiling, col)
    axes[-1].set_xlabel("time since first read (µs)")
    fig.suptitle("Active cores over time — the sub-saturation straggler tail stretches the envelope", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight")
    print(f"chart -> {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
