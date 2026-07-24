#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Two-panel timeline contrasting K=2 run two ways, from raw device-profiler CSVs.

  top panel  -- back-to-back: two separate program enqueues, each reader->compute->writer + end
                barrier, with the op-to-op sync (barrier + relaunch) BETWEEN them. The barrier
                RESYNCS all cores, so op1 starts fresh and the read/completion skew does NOT compound.
  bottom     -- unrolled: one program, workload run twice, NO barrier between reps. Cores never
                resync, so the fast core races through both reps while the slow core is starved and
                the fast/slow spread compounds.

Each panel draws the fastest and slowest core (by final done time) as two sub-tracks: reader (NoC0)
and writer (NoC1), on a shared absolute time axis. The generic event_timeline.py can't do this
(it keeps only the first occurrence of each marker per NCRISC_GO, dropping rep1's read burst).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_op_to_op_profiler_csv import load_profiler_csv, parse_chip_freq_mhz  # noqa: E402

T = "time[cycles since reset]"
WANT = {
    "NCRISC_GO",
    "READ_BEFORE_BARRIER",
    "READ_LAST_BARRIER",
    "WRITE_BEFORE_BARRIER",
    "WRITE_LAST_ISSUED",
    "WRITE_AFTER_BARRIER",
    "BRISC_DONE",
}
C_R0, C_R1, C_W, C_DR = "#66bb6a", "#2e7d32", "#1e88e5", "#e53935"


def per_core_invocations(df):
    """{(chip,cx,cy): [invocation,...]} in time order. invocation: go, read_bursts=[(before,last)],
    wbefore, wli, wafter, done."""
    m = df[df["zone name"].isin(WANT)]
    out = {}
    for key, g in m.groupby(["PCIe slot", "core_x", "core_y"], sort=False):
        evs = sorted((int(r[T]), r["zone name"]) for _, r in g.iterrows())
        invs, cur = [], None
        for t, name in evs:
            if name == "NCRISC_GO":
                if cur is not None:
                    invs.append(cur)
                cur = {"go": t, "rb": [], "rl": [], "wbefore": None, "wli": None, "wafter": None, "done": None}
            elif cur is not None:
                if name == "READ_BEFORE_BARRIER":
                    cur["rb"].append(t)
                elif name == "READ_LAST_BARRIER":
                    cur["rl"].append(t)
                elif name == "WRITE_BEFORE_BARRIER" and cur["wbefore"] is None:
                    cur["wbefore"] = t
                elif name == "WRITE_LAST_ISSUED":
                    cur["wli"] = t
                elif name == "WRITE_AFTER_BARRIER":
                    cur["wafter"] = t
                elif name == "BRISC_DONE":
                    cur["done"] = t
        if cur is not None:
            invs.append(cur)
        good = [v for v in invs if v["rb"] and v["rl"] and v["wafter"] is not None and v["wbefore"] is not None]
        for v in good:
            v["read_bursts"] = list(zip(sorted(v["rb"]), sorted(v["rl"])))
        out[key] = good
    return out


def draw_core(ax, ops, y, us, name):
    """Draw one core's reader (NoC0, upper) + writer (NoC1, lower) sub-tracks across its op(s)."""
    RH = 0.30
    yr, yw = y + 0.36, y
    burst_colors = [C_R0, C_R1]
    for op in ops:
        for (b, l), col in zip(op["read_bursts"], burst_colors):
            ax.broken_barh([(us(b), us(l) - us(b))], (yr, RH), facecolors=col, edgecolor="k", linewidth=0.3)
        ax.broken_barh(
            [(us(op["wbefore"]), us(op["wli"]) - us(op["wbefore"]))],
            (yw, RH),
            facecolors=C_W,
            edgecolor="k",
            linewidth=0.3,
        )
        ax.broken_barh(
            [(us(op["wli"]), us(op["wafter"]) - us(op["wli"]))], (yw, RH), facecolors=C_DR, edgecolor="k", linewidth=0.3
        )
    ax.text(-3, yr + RH / 2, "NoC0 rd", va="center", ha="right", fontsize=7, color="#2e7d32")
    ax.text(-3, yw + RH / 2, "NoC1 wr", va="center", ha="right", fontsize=7, color="#1565c0")
    last = max(us(op["wafter"]) for op in ops)
    ax.axvline(last, ls=":", color="#444", lw=0.9, alpha=0.6)
    ax.text(
        last + 2,
        y + 0.33,
        f"{name}\nfinish {last:.0f}µs",
        va="center",
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    return last


def panel(ax, df, freq, mode, title):
    inv = per_core_invocations(df)
    inv = {k: v for k, v in inv.items() if v}
    # ops used per core: unroll -> last invocation (2 read bursts); b2b -> last 2 invocations (op0,op1)
    sel = {}
    for k, v in inv.items():
        sel[k] = v[-2:] if mode == "b2b" else v[-1:]
    # normalize to the earliest go among the SELECTED (measured) ops, not all invocations
    t0 = min(ops[0]["go"] for ops in sel.values())
    us = lambda c: (c - t0) / freq
    donef = lambda ops: max(o["wafter"] for o in ops)
    fast_k = min(sel, key=lambda k: donef(sel[k]))
    slow_k = max(sel, key=lambda k: donef(sel[k]))
    ds = us(donef(sel[slow_k])) - us(donef(sel[fast_k]))
    draw_core(ax, sel[slow_k], 0.0, us, "slowest core")
    fast_end = draw_core(ax, sel[fast_k], 1.1, us, "fastest core")
    # done-spread bracket
    xf, xslo = us(donef(sel[fast_k])), us(donef(sel[slow_k]))
    ax.annotate("", xy=(xslo, 0.95), xytext=(xf, 0.95), arrowprops=dict(arrowstyle="<->", color="#6a1b9a", lw=1.3))
    ax.text((xf + xslo) / 2, 0.99, f"done spread {ds:.0f}us", ha="center", va="bottom", fontsize=8, color="#6a1b9a")
    if mode == "b2b":
        # mark op0 done -> op1 read start on the fast core (the resync gap)
        f = sel[fast_k]
        if len(f) == 2:
            gap0, gap1 = us(f[0]["wafter"]), us(f[1]["read_bursts"][0][0])
            ax.annotate(
                "", xy=(gap1, 1.48), xytext=(gap0, 1.48), arrowprops=dict(arrowstyle="<->", color="#ef6c00", lw=1.1)
            )
            ax.text(
                (gap0 + gap1) / 2,
                1.52,
                "barrier+relaunch\n(resync)",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#ef6c00",
            )
    ax.set_ylim(-0.25, 1.95)
    ax.set_yticks([])
    wall = us(donef(sel[slow_k]))  # K=2 wall-clock = slowest core's finish
    ax.set_title(f"{title}   |   K=2 wall-clock {wall:.0f}µs", fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.25)
    return ds, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2b-csv", type=Path, required=True)
    ap.add_argument("--unroll-csv", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    # Headline the CLEAN wall-clock gain from the REALTIME profiler (dispatch go/done, dump-free). The
    # device-profiler panels below are for per-core SHAPE only -- their op-to-op gap is inflated by the
    # ~3us L1->DRAM profiler dump and must NOT be read as the boundary cost. Pass the RT b2b/unroll walls
    # (medians) so the title reports the real gain; omit to fall back to the (perturbed) device wall.
    ap.add_argument("--rt-b2b-us", type=float, default=None)
    ap.add_argument("--rt-unroll-us", type=float, default=None)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    freq = parse_chip_freq_mhz(args.b2b_csv.resolve())
    b2b = load_profiler_csv(args.b2b_csv.resolve())
    unr = load_profiler_csv(args.unroll_csv.resolve())

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(13, 6.6), sharex=True)
    ds_b, wall_b = panel(axT, b2b, freq, "b2b", "back-to-back: 2 programs, barrier + resync between (K=2)")
    ds_u, wall_u = panel(axB, unr, freq, "unroll", "unrolled: 1 program, workload x2, NO mid-barrier (K=2)")
    axB.set_xlabel("time since first core go (us)")
    handles = [
        Patch(facecolor=C_R0, label="reads rep0/op0"),
        Patch(facecolor=C_R1, label="reads rep1/op1"),
        Patch(facecolor=C_W, label="writes"),
        Patch(facecolor=C_DR, label="end drain / barrier"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    if args.rt_b2b_us is not None and args.rt_unroll_us is not None:
        b_us, u_us, src = args.rt_b2b_us, args.rt_unroll_us, "realtime profiler"
    else:
        b_us, u_us, src = wall_b, wall_u, "device wall (profiler-perturbed)"
    gain = b_us - u_us
    pct = 100.0 * gain / b_us if b_us else 0.0
    fig.suptitle(
        f"K=2: back-to-back vs fused (no mid-barrier) — 56c, {b_us:.0f}µs op-pair.  "
        f"Fusing removes the boundary: {b_us:.0f}→{u_us:.0f}µs = {gain:.1f}µs ({pct:.0f}%) reclaimed [{src}].\n"
        f"Panels: device-profiler per-core shape — the op0→op1 gap shown includes the ~3µs L1→DRAM "
        f"profiler dump (measurement artifact; true dispatch gap ~0.1µs), so read it for SHAPE, not timing.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight")
    print(f"chart -> {args.out_png}")
    print(f"b2b done-spread={ds_b:.1f}us   unroll done-spread={ds_u:.1f}us")
    print(f"headline ({src}): b2b={b_us:.1f}us unroll={u_us:.1f}us gain={gain:.1f}us ({pct:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
