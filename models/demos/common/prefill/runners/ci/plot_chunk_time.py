# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import argparse
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scaling_summary import _REF, _load

_LINE = "#3987e5"
_BG, _INK, _MUTED, _GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#3a3a37"
_HALF_WINDOW = 2


def _probes(n):
    picked, d = [], 0
    while d < n - 1:
        picked.append(d)
        d = 2 * d + 1
    return picked + [n - 1]


def _value(curve, d):
    if d == 0:
        return curve[0]
    return statistics.median(curve[max(1, d - _HALF_WINDOW) : d + _HALF_WINDOW + 1])


def _label(tokens):
    return f"{round(tokens / 1048576)}M" if tokens >= 1048576 else f"{round(tokens / 1024)}K"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True, help="dir of *.json metrics sidecars (searched recursively)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    rec = _load(args.metrics_dir).get(args.model, {}).get(_REF, {})
    curve = rec.get("chunk_time_curve_ms")
    if not curve:
        print(f"no per-chunk {_REF} chunk_time for {args.model}; plot skipped")
        return 0

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=11)
    ax.grid(color=_GRID, lw=0.6)
    ax.set_xscale("log", base=2)
    ax.minorticks_off()

    picked = _probes(len(curve))
    xs = [(d + 1) * rec["chunk_size"] for d in picked]
    ys = [_value(curve, d) for d in picked]
    ax.plot(xs, ys, color=_LINE, lw=2.2, marker="o", ms=8, mfc=_BG, mec=_LINE, mew=2)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", color=_INK)

    ax.set_xticks(xs, [_label(x) for x in xs])
    ax.set_ylim(0, max(ys) * 1.18)
    ax.set_xlabel("Input sequence length (tokens, log₂ scale)", color=_MUTED, fontsize=12)
    ax.set_ylabel("chunk_time (ms), lower is better", color=_MUTED, fontsize=12)
    ax.set_title(args.title or f"{args.model} {_REF} chunk_time@ISL", color=_INK, fontsize=13, loc="left")
    fig.text(
        0.01,
        0.01,
        "chunk_time = a chunk's first layer ack -> its last layer ack. First point is chunk 0; "
        f"the rest are the median of the {2 * _HALF_WINDOW + 1} chunks around that ISL.",
        color=_MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
