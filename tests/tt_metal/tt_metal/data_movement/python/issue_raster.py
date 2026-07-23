# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-core issue-raster plot for the multi_interleaved reader/writer.

When the reader or writer kernel is built/run with TT_DM_PAGE_COUNTERS=1 it emits one
"dm_page_issued" TS_DATA marker right after every page transaction on the batched (!sync)
path. Each marker carries:

    time[cycles since reset] - the cycle the transaction was issued
    data                     - cumulative returns so far (reads: NIU_MST_RD_RESP_RECEIVED;
                               non-posted writes: NIU_MST_WR_ACK_RECEIVED; posted writes:
                               NIU_MST_POSTED_WR_REQ_SENT)

This script renders, for one (test_id, Q, transaction_size) point, a raster where:

    y-axis = core index (1 at the bottom .. num_cores at the top), matching the host's
             corerange_to_cores enumeration (row_wise=false -> column-major: index = x_rank*num_y + y_rank)
    x-axis = time in cycles (normalized so the earliest issue across all cores is t=0)
    point  = one issued read, colored by responses-so-far

The horizontal spacing between a core's points is its per-page issue cadence; a widening gap
where the color (responses) keeps climbing is noc_cmd_buf_ready backpressure - the NIU stalling
issue until an outstanding read frees a command-buffer slot.
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from tracy.process_device_log import extract_device_info  # noqa: F401 (kept for arch parity/side effects)
from tests.tt_metal.tt_metal.data_movement.python.constants import DEFAULT_OUTPUT_DIR
from loguru import logger

PAGE_ZONE = "dm_page_issued"
META_ZONES = {"Test id", "Number of transactions", "Transaction size in bytes"}


def parse_csv(file_path):
    """Return (meta_by_run, issues_by_run).

    meta_by_run[rhid][zone] = payload int
    issues_by_run[rhid][(x, y)] = list of (timestamp, responses_so_far)
    """
    meta_by_run = {}
    issues_by_run = defaultdict(lambda: defaultdict(list))
    with open(file_path, "r") as f:
        f.readline()  # skip the "ARCH: ..." banner line
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            if row.get("type") != "TS_DATA":
                continue
            zone = row.get("zone name")
            if zone is None:
                continue
            try:
                rhid = int(row["run host ID"])
                payload = int(row["data"])
            except (TypeError, ValueError):
                continue
            if zone in META_ZONES:
                meta_by_run.setdefault(rhid, {})[zone] = payload
            elif zone == PAGE_ZONE:
                ts = int(row["time[cycles since reset]"])
                x = int(row["core_x"])
                y = int(row["core_y"])
                issues_by_run[rhid][(x, y)].append((ts, payload))
    return meta_by_run, issues_by_run


def select_run(meta_by_run, test_id, num_transactions, transaction_size):
    matches = [
        rhid
        for rhid, m in meta_by_run.items()
        if (test_id is None or m.get("Test id") == test_id)
        and (num_transactions is None or m.get("Number of transactions") == num_transactions)
        and (transaction_size is None or m.get("Transaction size in bytes") == transaction_size)
    ]
    if not matches:
        raise ValueError(f"No run matched test_id={test_id}, Q={num_transactions}, size={transaction_size}.")
    if len(matches) > 1:
        logger.warning(f"Multiple runs matched {matches}; using the newest (max run host ID).")
    return max(matches)


def core_index_map(cores):
    """Map each (x, y) to the host linear core index (0-based).

    corerange_to_cores defaults to row_wise=false, i.e. column-major over the logical grid:
        for x: for y: emit (x, y)  ->  index = x_rank * num_y + y_rank
    Translated NoC coordinates preserve logical order, so ranking the distinct x/y values
    reproduces the enumeration without needing to open the device.
    """
    xs = sorted({x for x, _ in cores})
    ys = sorted({y for _, y in cores})
    x_rank = {x: i for i, x in enumerate(xs)}
    y_rank = {y: i for i, y in enumerate(ys)}
    num_y = len(ys)
    return {(x, y): x_rank[x] * num_y + y_rank[y] for (x, y) in cores}, len(xs), num_y


def plot_one(issues_by_core, q, transaction_size, out_path, max_cores=None):
    idx_of, num_x, num_y = core_index_map(issues_by_core.keys())
    num_cores = num_x * num_y
    shown = num_cores if not max_cores else min(max_cores, num_cores)

    # Normalize time so the earliest issue across all cores is t=0 (over the shown cores).
    t_min = min(ts for core, pts in issues_by_core.items() if idx_of[core] < shown for ts, _ in pts)

    xs_t, ys_c, cvals = [], [], []
    per_core_counts = []
    for core, pts in issues_by_core.items():
        ci = idx_of[core]
        if ci >= shown:
            continue
        per_core_counts.append(len(pts))
        for ts, resp in pts:
            xs_t.append(ts - t_min)
            ys_c.append(ci + 1)  # 1-based: core 1 at the bottom
            cvals.append(resp)

    fig, ax = plt.subplots(figsize=(16, 10))
    sc = ax.scatter(xs_t, ys_c, c=cvals, cmap="viridis", s=8, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("read responses received so far (backpressure gauge)")

    ax.set_xlabel("time [cycles since first issue]")
    ax.set_ylabel("core index (1 = bottom, host corerange_to_cores order)")
    ax.set_ylim(0.5, shown + 0.5)
    ax.set_title(
        f"Per-core issue raster - Q={q}, {transaction_size} B, first {shown}/{num_cores} cores "
        f"({num_x}x{num_y}), points/core={min(per_core_counts)}..{max(per_core_counts)}"
    )
    ax.grid(True, axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}  (cores={num_cores}, issues={len(xs_t)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-core issue-raster (time vs core) for multi_interleaved reads")
    parser.add_argument("-l", "--log_csv", default="generated/profiler/.logs/profile_log_device.csv")
    parser.add_argument("-i", "--test_id", type=int, default=129)
    parser.add_argument("-s", "--transaction_size", type=int, default=4096)
    parser.add_argument(
        "-q", "--num_transactions", type=int, nargs="+", default=[4, 16, 64], help="Q value(s); one figure per Q"
    )
    parser.add_argument(
        "-c", "--max_cores", type=int, default=55, help="Limit the y-axis to the first N cores (0 = all)"
    )
    parser.add_argument("-o", "--output_dir", default=None)
    args = parser.parse_args()

    meta_by_run, issues_by_run = parse_csv(args.log_csv)
    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_DIR, "multi_interleaved", "read_grid_sweep")
    os.makedirs(output_dir, exist_ok=True)

    for q in args.num_transactions:
        rhid = select_run(meta_by_run, args.test_id, q, args.transaction_size)
        if rhid not in issues_by_run or not issues_by_run[rhid]:
            raise ValueError(f"Run {rhid} (Q={q}) has no '{PAGE_ZONE}' markers. Re-run with TT_DM_PAGE_COUNTERS=1.")
        out_path = os.path.join(output_dir, f"issue_raster_{args.test_id}_{q}_{args.transaction_size}.png")
        plot_one(issues_by_run[rhid], q, args.transaction_size, out_path, max_cores=args.max_cores or None)
