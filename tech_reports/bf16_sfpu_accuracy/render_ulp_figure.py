#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Render the exhaustive BF16 pure-ULP comparison figure for ttnn.erfinv.

Inputs are the per-encoding .npz dumps produced by running
tests/ttnn/unit_tests/operations/eltwise/test_erfinv_bf16_exhaustive.py with
TT_EXPORT_ULP_DUMP=<path> on the stock and the patched tree (each dump holds
x, out and the float64 golden for all 65,536 BF16 encodings).

Usage:
    python render_ulp_figure.py --candidate cand.npz --baseline stock.npz         --out images/erfinv_bf16_ulp.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def pure_ulp(out: np.ndarray, golden: np.ndarray) -> np.ndarray:
    """ttnn-eltwise-op-tester metric: |FTZ(golden) - out| / bf16_ulp(rounded golden).

    The rounded golden gets post-round FTZ (Blackhole flushes after rounding)
    and the numerator flush is keyed strictly on the rounded golden: a golden
    in the top half-ULP below MIN_NORMAL rounds up onto MIN_NORMAL and is not
    flushed.
    """
    finite = np.isfinite(golden)
    y = golden[finite]
    rounded = torch.from_numpy(y).to(torch.bfloat16).to(torch.float64).numpy()
    sub = (np.abs(rounded) < 2.0**-126) & (rounded != 0.0)
    rounded = np.where(sub, np.copysign(0.0, rounded), rounded)
    y32 = np.abs(rounded.astype(np.float32))
    bits = (y32.view(np.uint32) >> np.uint32(16)).astype(np.uint32)
    nxt = np.minimum(bits + np.uint32(1), np.uint32(0x7F80))
    spacing = (nxt << np.uint32(16)).view(np.float32) - (bits << np.uint32(16)).view(np.float32)
    y_ftz = np.where(rounded == 0.0, np.copysign(0.0, y), y)
    ulp = np.full(golden.shape, np.nan)
    ulp[finite] = (np.abs(y_ftz - out[finite]).astype(np.float32) / spacing.astype(np.float32)).astype(np.float64)
    return ulp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cand = np.load(args.candidate)
    base = np.load(args.baseline)
    x, golden = cand["x"], cand["golden"]
    assert np.array_equal(x, base["x"], equal_nan=True), "dumps cover different inputs"

    ulp_cand = pure_ulp(cand["out"], golden)
    ulp_base = pure_ulp(base["out"], golden)

    finite = np.isfinite(golden)
    order = np.argsort(x[finite], kind="stable")
    xs = x[finite][order]
    # Exact results are clamped to 1e-6 so they stay visible on the log axis.
    yc = np.maximum(ulp_cand[finite][order], 1e-6)
    yb = np.maximum(ulp_base[finite][order], 1e-6)

    surface, ink, muted = "#fcfcfb", "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.scatter(
        xs,
        yb,
        s=1.5,
        color="#eb6834",
        alpha=0.55,
        linewidths=0,
        label=f"previous ttnn.erfinv (max {np.nanmax(ulp_base):.1f} ULP)",
    )
    ax.scatter(
        xs,
        yc,
        s=1.5,
        color="#2a78d6",
        alpha=0.7,
        linewidths=0,
        label=f"replacement (max {np.nanmax(ulp_cand):.3f} ULP)",
    )
    ax.axhline(1.0, color=muted, lw=1.2, ls=(0, (4, 3)))
    ax.annotate(
        "1 ULP", xy=(xs[-1], 1.0), xytext=(-4, 5), textcoords="offset points", ha="right", color=muted, fontsize=9
    )
    ax.set_yscale("log")
    ax.set_xlabel("input x (every finite-domain BF16 encoding, ordered by value)", color=ink)
    ax.set_ylabel("pure ULP error (log scale)", color=ink)
    ax.set_title(
        f"ttnn.erfinv BF16 accuracy, exhaustive on Blackhole silicon (all 65,536 encodings)",
        color=ink,
        fontsize=12,
    )
    legend = ax.legend(loc="upper left", framealpha=0.95, edgecolor="#d8d7d2")
    for text in legend.get_texts():
        text.set_color(ink)
    ax.tick_params(colors=muted)
    for spine in ax.spines.values():
        spine.set_color("#d8d7d2")
    ax.grid(True, which="major", color="#e4e3de", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=surface)
    print(f"wrote {args.out}")
    print(f"candidate max pure ULP: {np.nanmax(ulp_cand):.6f}")
    print(f"baseline  max pure ULP: {np.nanmax(ulp_base):.6f}")


if __name__ == "__main__":
    main()
