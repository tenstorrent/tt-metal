#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Robust bandwidth-over-time / roll-off charts from measured per-core progress markers.

Uses READ_PROG / WRITE_PROG markers (emitted by the reader/writer kernels every N pages via
--read/--write-progress-every) to compute delivered/issued BW over time WITHOUT the per-interval
Δpages/Δt division, which blows up when two markers are microseconds apart. Instead we build each
core's monotonic cumulative-pages(t), sum across cores, and take a smoothed slope -- robust, no spikes.

Modes:
  alloc   : one config; plot fast-half vs slow-half vs total BW over time (split by finish time).
  compare : several configs (e.g. 8c, 56c); one panel each, active cores (left) + total BW (right),
            with the roll-off onset annotated on the highest-core-count panel.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_op_to_op_profiler_csv import load_profiler_csv, parse_chip_freq_mhz  # noqa: E402
from chart_timeline_k2 import per_core_invocations  # noqa: E402

BYTES_PER_PAGE = 2048


def per_core(csv, marker, anchor_first=False):
    """Per core: cumulative (t,pages) samples anchored at phase start, + finish time. freq/t0/env too.

    anchor_first=True moves each core's zero-anchor from the EV_WRITE_BEFORE/first-read-burst timestamp
    (which precedes the first actual page by the pipeline-fill delay -> dilutes early slope into a fake
    ramp) to the BACK-PROJECTED write-start: one marker-interval before the first marker, at the steady
    rate. That spreads the first chunk of pages over real time instead of leaving it at wbefore (ramp)
    or crediting it instantly at the first marker (a fake startup spike).
    """
    freq = parse_chip_freq_mhz(Path(csv).resolve())
    df = load_profiler_csv(Path(csv).resolve())
    inv = {k: v for k, v in per_core_invocations(df).items() if v}
    start, finish = {}, {}
    for (s, x, y), v in inv.items():
        op = v[-1]
        if marker == "WRITE_PROG":
            start[(x, y)], finish[(x, y)] = op["wbefore"], op["wafter"]
        else:  # READ_PROG
            if not op["read_bursts"]:
                continue
            start[(x, y)], finish[(x, y)] = op["read_bursts"][0][0], op["read_bursts"][-1][1]
    T = df[df["zone name"] == marker]
    samp = {}
    for (x, y), g in T.groupby(["core_x", "core_y"]):
        rows = sorted((int(t), int(d)) for t, d in zip(g["time[cycles since reset]"], g["data"]))
        mn = min(d for _, d in rows)
        idx = [i for i, (t, d) in enumerate(rows) if d == mn]
        last = rows[idx[-1] :]  # last program instance
        if (x, y) in start:
            if anchor_first and len(last) >= 2:
                # Zero at the back-projected write-start: t1 - (t2 - t1), so the first chunk of pages is
                # spread over one marker-interval at the steady rate (no wbefore ramp, no zero-time spike).
                (t1, _), (t2, _) = last[0], last[1]
                samp[(x, y)] = [(2 * t1 - t2, 0)] + last
            else:
                samp[(x, y)] = [(start[(x, y)], 0)] + last
    t0 = min(s[0][0] for s in samp.values())
    env = max(finish[k] for k in samp)
    return samp, finish, freq, t0, (env - t0) / freq


def cum_at(s, t, us):
    if t <= us(s[0][0]):
        return 0.0
    if t >= us(s[-1][0]):
        return s[-1][1]
    for i in range(1, len(s)):
        a, b = us(s[i - 1][0]), us(s[i][0])
        if t <= b:
            return s[i - 1][1] + (s[i][1] - s[i - 1][1]) * (t - a) / (b - a)
    return s[-1][1]


def bw_curve(samp, cores, t0, freq, env, ts, smooth_us):
    """Robust BW(t) for a set of cores: cumulative pages -> bytes, smoothed slope over +/- smooth_us."""
    import numpy as np

    us = lambda c: (c - t0) / freq
    cum = np.array([sum(cum_at(samp[k], t, us) for k in cores) * BYTES_PER_PAGE for t in ts])
    w = max(1, int(smooth_us / (env / len(ts))))
    r = np.zeros(len(ts))
    for i in range(len(ts)):
        lo, hi = max(0, i - w), min(len(ts) - 1, i + w)
        r[i] = (cum[hi] - cum[lo]) / ((ts[hi] - ts[lo]) * 1e3) if ts[hi] > ts[lo] else 0.0
    return r


def active_count(samp, t0, freq, ts):
    us = lambda c: (c - t0) / freq
    return [sum(1 for s in samp.values() if us(s[0][0]) <= t <= us(s[-1][0])) for t in ts]


def main():
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--marker", default="WRITE_PROG", choices=["READ_PROG", "WRITE_PROG"])
    ap.add_argument("--mode", required=True, choices=["alloc", "compare"])
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--smooth-us", type=float, default=5.0)
    ap.add_argument(
        "--anchor-first",
        action="store_true",
        help="anchor cumulative at first real marker (removes fill-dilution ramp artifact)",
    )
    args = ap.parse_args()
    rw = "read" if args.marker == "READ_PROG" else "write"
    noun = {"read": "readers", "write": "writers"}[rw]  # correct plural (avoid "writeers")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.mode == "alloc":
        samp, finish, freq, t0, env = per_core(args.csv[0], args.marker, args.anchor_first)
        order = sorted(samp, key=lambda k: finish[k])
        h = len(order) // 2
        ts = np.linspace(0, env, 200)
        fast = bw_curve(samp, order[:h], t0, freq, env, ts, args.smooth_us)
        slow = bw_curve(samp, order[h:], t0, freq, env, ts, args.smooth_us)
        tot = fast + slow
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ts, tot, color="k", lw=2.5, label="TOTAL")
        ax.plot(ts, fast, color="#2e7d32", lw=2, label=f"fast {noun} ({h}, finish first)")
        ax.plot(ts, slow, color="#c62828", lw=2, label=f"slow {noun} ({len(order) - h})")
        avg = sum(samp[k][-1][1] for k in samp) * BYTES_PER_PAGE / env / 1e3
        ax.axhline(avg, ls=":", color="k", lw=1.5)
        ax.text(env * 0.98, avg + 2, f"avg over full run = {avg:.0f} GB/s", ha="right", va="bottom", fontsize=8)
        ax.set_xlabel(f"time since first {rw} (µs)")
        ax.set_ylabel(f"measured {rw} BW (GB/s)")
        ax.set_title(args.title or f"{args.label[0]} {rw} BW allocation over time (measured, cumulative)", fontsize=11)
        ax.legend()
        ax.grid(alpha=0.3)
    else:  # compare
        n = len(args.csv)
        # Gather all configs first: we need each config's bulk BW and the cross-config cumulative
        # crossover to annotate the PRIMARY cause (lower sustained BW), not just the roll-off tail.
        cfgs = []
        for csv, lab in zip(args.csv, args.label):
            samp, finish, freq, t0, env = per_core(csv, args.marker, args.anchor_first)
            ts = np.linspace(0, env, 200)
            us = lambda c, t0=t0, freq=freq: (c - t0) / freq
            bw = bw_curve(samp, list(samp), t0, freq, env, ts, args.smooth_us)
            cum = np.array([sum(cum_at(samp[k], t, us) for k in samp) * BYTES_PER_PAGE for t in ts])
            act = active_count(samp, t0, freq, ts)
            mask = (ts > 0.2 * env) & (ts < 0.7 * env)  # steady state (excl. fill + roll-off)
            cfgs.append(
                dict(lab=lab, nc=len(samp), ts=ts, bw=bw, cum=cum, act=act, env=env, bulk=float(np.median(bw[mask])))
            )
        lo = min(cfgs, key=lambda c: c["nc"])
        hi = max(cfgs, key=lambda c: c["nc"])
        cross = None
        if hi is not lo:  # when does hi's cumulative fall behind lo's (same total work)?
            lo_on_hi = np.interp(hi["ts"], lo["ts"], lo["cum"])
            cross = next(
                (hi["ts"][i] for i in range(len(hi["ts"])) if hi["cum"][i] < lo_on_hi[i] - 0.02 * lo["cum"][-1]), None
            )
        fig, axes = plt.subplots(n, 1, figsize=(12, 3.6 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, c in zip(axes, cfgs):
            ax.fill_between(c["ts"], c["act"], step="mid", color="#1e88e5", alpha=0.20)
            ax.plot(c["ts"], c["act"], color="#1e88e5", lw=1.3, drawstyle="steps-mid")
            ax.set_ylabel(f"active {noun}", color="#1e88e5")
            ax.tick_params(axis="y", labelcolor="#1e88e5")
            ax.set_ylim(0, c["nc"] * 1.08)
            a2 = ax.twinx()
            a2.plot(c["ts"], c["bw"], color="#e65100", lw=2.2)
            a2.set_ylabel(f"measured {rw} BW (GB/s)", color="#e65100")
            a2.tick_params(axis="y", labelcolor="#e65100")
            pk = max(float(c["bw"].max()), hi["bulk"], lo["bulk"])
            a2.set_ylim(0, pk * 1.18)
            # PRIMARY story: this config's sustained (bulk) BW plateau, dotted
            a2.axhline(c["bulk"], ls=":", color="#e65100", lw=1.6)
            a2.text(
                c["env"] * 0.5,
                c["bulk"] + pk * 0.03,
                f"sustained BW {c['bulk']:.0f} GB/s",
                color="#bf360c",
                fontsize=9,
                ha="center",
                fontweight="bold",
            )
            a2.axvline(c["env"], ls=":", color="#555", lw=1.1)
            a2.text(c["env"], pk * 1.05, f"done {c['env']:.0f}µs", fontsize=8, ha="right")
            ax.set_title(f"{c['lab']}: {c['nc']} cores, envelope {c['env']:.0f}µs", loc="left", fontsize=11)
            ax.grid(alpha=0.25)
            if c is hi and hi is not lo:
                # overlay lo's sustained level + mark where hi falls behind; decompose the gap
                a2.axhline(lo["bulk"], ls=":", color="#2e7d32", lw=1.4)
                a2.text(
                    c["env"] * 0.5,
                    lo["bulk"] + pk * 0.03,
                    f"{lo['lab']} sustained {lo['bulk']:.0f} GB/s",
                    color="#2e7d32",
                    fontsize=9,
                    ha="center",
                )
                noroll = c["cum"][-1] / (c["bulk"] * 1e3)  # env if it held bulk BW the whole way (no roll-off)
                low_bw_us = noroll - lo["env"]
                roll_us = c["env"] - noroll
                if cross is not None:
                    a2.axvline(cross, ls="--", color="#6a1b9a", lw=1.4)
                    a2.text(
                        cross,
                        pk * 0.92,
                        f"falls behind {lo['lab']}\nat t≈{cross:.0f}µs",
                        color="#6a1b9a",
                        fontsize=8,
                        ha="left",
                    )
                a2.annotate(
                    f"PRIMARY: sustained BW {c['bulk']:.0f} < {lo['lab']}'s {lo['bulk']:.0f} "
                    f"({100 * (lo['bulk'] - c['bulk']) / lo['bulk']:.0f}% lower) → +{low_bw_us:.0f}µs\n"
                    f"SECONDARY: end roll-off tail → +{roll_us:.0f}µs",
                    xy=(c["env"] * 0.5, c["bulk"]),
                    xytext=(c["env"] * 0.04, pk * 0.30),
                    fontsize=9,
                    color="#bf360c",
                    fontweight="bold",
                )
        axes[-1].set_xlabel(f"time since first {rw} (µs)")
        fig.suptitle(
            args.title or f"{rw.capitalize()} roll-off (measured BW): active cores vs delivered BW",
            fontsize=12,
            fontweight="bold",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97] if args.mode == "compare" else None)
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight")
    print(f"chart -> {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
