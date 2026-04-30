#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline heatmap plotter for MoE expert activation CSV logs.

Consumes the CSV emitted by
``tt-train/sources/examples/nano_gpt/moe_activation_logger.py``
(``step,layer,expert,prob``) and renders one heatmap per MoE layer
(x = logged step, y = expert, color = P(activation) in [0, 1]).

Example::

    python tt-train/scripts/plot_expert_activation.py \\
        --input moe_activation.csv \\
        --output moe_activation.png

    # Single-layer output
    python tt-train/scripts/plot_expert_activation.py \\
        --input moe_activation.csv --output layer0.png --layer 0

    # Custom colormap
    python tt-train/scripts/plot_expert_activation.py \\
        --input moe_activation.csv --output moe_activation.png --cmap plasma
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless-friendly; safe even with a display present.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to activation CSV.")
    parser.add_argument("--output", required=True, help="Destination PNG.")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="If set, plot only this MoE layer index; otherwise plot all.",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap name (default: viridis).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default: 150).",
    )
    return parser.parse_args()


def _pivot_layer(df_layer: pd.DataFrame) -> tuple[np.ndarray, list[int], list[int]]:
    """Return ``(matrix, steps, experts)`` where ``matrix[e, s]`` is the
    activation probability of expert ``experts[e]`` at ``steps[s]``.
    """
    pivot = df_layer.pivot_table(index="expert", columns="step", values="prob", aggfunc="mean")
    # Make sure we have a dense contiguous grid in expert order and step order.
    experts = sorted(pivot.index.tolist())
    steps = sorted(pivot.columns.tolist())
    pivot = pivot.reindex(index=experts, columns=steps)
    return pivot.to_numpy(dtype=np.float32), steps, experts


def _plot_layer(
    ax,
    matrix: np.ndarray,
    steps: list[int],
    experts: list[int],
    layer_idx: int,
    cmap: str,
) -> None:
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    num_steps = len(steps)
    num_experts = len(experts)

    # Keep tick labels readable even when the step list is long.
    max_xticks = 16
    if num_steps <= max_xticks:
        xtick_idx = list(range(num_steps))
    else:
        xtick_idx = np.linspace(0, num_steps - 1, max_xticks, dtype=int).tolist()
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels([str(steps[i]) for i in xtick_idx], rotation=45, ha="right", fontsize=8)

    max_yticks = 32
    if num_experts <= max_yticks:
        ytick_idx = list(range(num_experts))
    else:
        ytick_idx = np.linspace(0, num_experts - 1, max_yticks, dtype=int).tolist()
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels([str(experts[i]) for i in ytick_idx], fontsize=8)

    ax.set_xlabel("step")
    ax.set_ylabel("expert")
    uniform = 1.0 / max(num_experts, 1)  # purely informational
    ax.set_title(
        f"MoE layer {layer_idx}  (uniform target = n_activated/num_experts)\n"
        f"fully-balanced lower bound ~ {uniform:.3f}",
        fontsize=9,
    )
    plt.colorbar(im, ax=ax, label="P(activation)")


def main() -> int:
    args = _parse_args()

    if not os.path.exists(args.input):
        print(f"error: input CSV not found: {args.input}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input)
    required_cols = {"step", "layer", "expert", "prob"}
    if not required_cols.issubset(df.columns):
        print(f"error: missing columns {required_cols - set(df.columns)} in {args.input}", file=sys.stderr)
        return 1

    if args.layer is not None:
        df = df[df["layer"] == args.layer]
        if df.empty:
            print(f"error: no rows with layer == {args.layer}", file=sys.stderr)
            return 1

    layer_ids = sorted(df["layer"].unique().tolist())
    if not layer_ids:
        print(f"error: no data rows in {args.input}", file=sys.stderr)
        return 1

    n = len(layer_ids)
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, layer_id in enumerate(layer_ids):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        matrix, steps, experts = _pivot_layer(df[df["layer"] == layer_id])
        _plot_layer(ax, matrix, steps, experts, layer_id, args.cmap)

    # Hide any unused subplot slots.
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("MoE expert activation probability per training step", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}  ({n} layer(s), {df.shape[0]} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
