# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-core issue-progress curves for the multi_interleaved reader.

Consumes the "dm_page_issued" TS_DATA markers (TT_DM_PAGE_COUNTERS=1). For each core it draws a
monotonic staircase of "reads issued so far" vs time:

    x = time [cycles since the first issue across the shown cores]
    y = cumulative reads issued (0 .. Q)

A steep line = the core is issuing at full rate; a line that flattens = the core is stalled on
noc_cmd_buf_ready backpressure (issue paced by returning responses). Lines are colored by the
core's column rank (DRAM-distance proxy) so the fast edge-columns vs slow center-columns separate.

This is the "envelope" view of the raster: instead of one dot per read, you see each core's whole
issue trajectory, and the spread between the fastest and slowest curve is the injection skew.
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from tests.tt_metal.tt_metal.data_movement.python.issue_raster import parse_csv, select_run, core_index_map
from tests.tt_metal.tt_metal.data_movement.python.constants import DEFAULT_OUTPUT_DIR
from loguru import logger


def plot_one(issues_by_core, q, transaction_size, out_path, max_cores=None):
    idx_of, num_x, num_y = core_index_map(issues_by_core.keys())
    num_cores = num_x * num_y
    shown = num_cores if not max_cores else min(max_cores, num_cores)

    t_min = min(ts for core, pts in issues_by_core.items() if idx_of[core] < shown for ts, _ in pts)

    # Color each core by its column rank (x position) = DRAM-distance proxy.
    cmap = cm.get_cmap("turbo")
    norm = Normalize(vmin=0, vmax=max(num_x - 1, 1))

    fig, ax = plt.subplots(figsize=(15, 9))
    for core, pts in issues_by_core.items():
        ci = idx_of[core]
        if ci >= shown:
            continue
        col_rank = ci // num_y  # x_rank (column-major index)
        ts = np.array(sorted(t - t_min for t, _ in pts))
        y = np.arange(1, len(ts) + 1)
        # Step from (t_k, k) so a horizontal run is a stall (no issue progress).
        ax.step(ts, y, where="post", color=cmap(norm(col_rank)), alpha=0.55, linewidth=1.0)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("core column rank (0 = one DRAM edge .. far = center/other edge)")

    ax.set_xlabel("time [cycles since first issue]")
    ax.set_ylabel("cumulative reads issued")
    ax.set_ylim(0, q)
    ax.set_title(
        f"Per-core issue progress - Q={q}, {transaction_size} B, first {shown}/{num_cores} cores "
        f"({num_x}x{num_y}); flat segment = noc_cmd_buf_ready stall"
    )
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}  (cores shown={shown})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-core issue-progress curves for multi_interleaved reads")
    parser.add_argument("-l", "--log_csv", default="generated/profiler/.logs/profile_log_device.csv")
    parser.add_argument("-i", "--test_id", type=int, default=129)
    parser.add_argument("-s", "--transaction_size", type=int, default=4096)
    parser.add_argument("-q", "--num_transactions", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("-c", "--max_cores", type=int, default=55, help="Plot the first N cores (0 = all)")
    parser.add_argument("-o", "--output_dir", default=None)
    args = parser.parse_args()

    meta_by_run, issues_by_run = parse_csv(args.log_csv)
    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_DIR, "multi_interleaved", "read_grid_sweep")
    os.makedirs(output_dir, exist_ok=True)

    for q in args.num_transactions:
        rhid = select_run(meta_by_run, args.test_id, q, args.transaction_size)
        if rhid not in issues_by_run or not issues_by_run[rhid]:
            raise ValueError(f"Run {rhid} (Q={q}) has no page markers. Re-run with TT_DM_PAGE_COUNTERS=1.")
        out_path = os.path.join(output_dir, f"issue_progress_{args.test_id}_{q}_{args.transaction_size}.png")
        plot_one(issues_by_run[rhid], q, args.transaction_size, out_path, max_cores=args.max_cores or None)
