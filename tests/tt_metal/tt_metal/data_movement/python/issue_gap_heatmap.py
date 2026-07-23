# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-core issue-gap heatmap for the multi_interleaved reader.

Consumes the "dm_page_issued" TS_DATA markers (TT_DM_PAGE_COUNTERS=1) and renders a 2D map:

    y = core index (1 at the bottom .. num_cores at the top), host corerange_to_cores order
    x = page index within the batch (0 .. Q-1)
    color = the issue gap before that page = t[page] - t[page-1]  [cycles]

A row that is uniformly dark = that core injects at a steady cadence (no stalls). A row that
turns bright partway across = the noc_cmd_buf_ready knee: once ~a window's worth of reads are
outstanding, each further issue waits for a response, so the per-page gap jumps. Comparing Q=16
(no wall) vs Q=64 (a bright wall appears mid-batch) makes the finite outstanding window obvious.

Color is clipped at a robust percentile so a few huge outliers don't wash out the structure.
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tests.tt_metal.tt_metal.data_movement.python.issue_raster import parse_csv, select_run, core_index_map
from tests.tt_metal.tt_metal.data_movement.python.constants import DEFAULT_OUTPUT_DIR
from loguru import logger


def plot_one(issues_by_core, q, transaction_size, out_path, max_cores=None, clip_pct=98.0):
    idx_of, num_x, num_y = core_index_map(issues_by_core.keys())
    num_cores = num_x * num_y
    shown = num_cores if not max_cores else min(max_cores, num_cores)

    # gaps[core_index, page] ; page 0 has no predecessor -> NaN
    gaps = np.full((shown, q), np.nan)
    for core, pts in issues_by_core.items():
        ci = idx_of[core]
        if ci >= shown:
            continue
        ts = np.array(sorted(t for t, _ in pts))
        d = np.diff(ts)  # length len(ts)-1, gap before page 1..len-1
        n = min(len(d), q - 1)
        gaps[ci, 1 : 1 + n] = d[:n]

    finite = gaps[np.isfinite(gaps)]
    vmax = np.percentile(finite, clip_pct) if finite.size else 1.0
    vmin = finite.min() if finite.size else 0.0

    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#303030")
    im = ax.imshow(
        gaps, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, extent=[-0.5, q - 0.5, 0.5, shown + 0.5]
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"issue gap t[p]-t[p-1] [cycles] (clipped at p{clip_pct:g})")

    ax.set_xlabel("page index within batch")
    ax.set_ylabel("core index (1 = bottom, host corerange_to_cores order)")
    ax.set_title(
        f"Per-core issue-gap heatmap - Q={q}, {transaction_size} B, first {shown}/{num_cores} cores "
        f"({num_x}x{num_y}); bright column band = noc_cmd_buf_ready knee"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}  (cores shown={shown}, vmax@p{clip_pct:g}={vmax:.0f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-core issue-gap heatmap for multi_interleaved reads")
    parser.add_argument("-l", "--log_csv", default="generated/profiler/.logs/profile_log_device.csv")
    parser.add_argument("-i", "--test_id", type=int, default=129)
    parser.add_argument("-s", "--transaction_size", type=int, default=4096)
    parser.add_argument("-q", "--num_transactions", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("-c", "--max_cores", type=int, default=0, help="Show the first N cores (0 = all)")
    parser.add_argument("-o", "--output_dir", default=None)
    args = parser.parse_args()

    meta_by_run, issues_by_run = parse_csv(args.log_csv)
    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_DIR, "multi_interleaved", "read_grid_sweep")
    os.makedirs(output_dir, exist_ok=True)

    for q in args.num_transactions:
        rhid = select_run(meta_by_run, args.test_id, q, args.transaction_size)
        if rhid not in issues_by_run or not issues_by_run[rhid]:
            raise ValueError(f"Run {rhid} (Q={q}) has no page markers. Re-run with TT_DM_PAGE_COUNTERS=1.")
        out_path = os.path.join(output_dir, f"issue_gap_heatmap_{args.test_id}_{q}_{args.transaction_size}.png")
        plot_one(issues_by_run[rhid], q, args.transaction_size, out_path, max_cores=args.max_cores or None)
