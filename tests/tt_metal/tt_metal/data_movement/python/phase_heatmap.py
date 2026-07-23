# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-core phase decomposition heatmaps for the data-movement reader/writer kernels.

The reader and writer kernels, when built with TT_DM_PHASE_COUNTERS=1, emit four
generic TS_DATA markers on the batched (!sync) path (same names for both, so one tool
handles read and write runs):

    dm_t0_issue_start   - issue loop start
    dm_t1_issue_end     - all transactions issued  (payload = returns in so far)
    dm_t2_first_return  - first return observed     (payload = returns in so far)
    dm_t3_barrier_clear - completion barrier cleared (payload = total returns)

"return" is kernel-specific: for reads it is a completed read response
(NIU_MST_RD_RESP_RECEIVED); for non-posted writes it is a write ack
(NIU_MST_WR_ACK_RECEIVED); for posted writes there is no ack, so it is a posted
request departing (NIU_MST_POSTED_WR_REQ_SENT).

Each marker carries BOTH a timestamp (col "time[cycles since reset]") and a payload
(col "data"). stats_collector keeps only the payload, so this script parses the raw
profiler CSV directly to recover the timestamps and build per-core interval maps:

    issue       = t1 - t0   (time to push all transaction commands out)
    first_return = t2 - t1   (wait for first completed return, whole page, not first byte;
                             ~0 when returns already came back during issue)
    drain       = t3 - t2   (first return -> last return; return-bandwidth bound.
                             ~overhead-only at Q=1: the single return is already in at t2)
    t1_payload  = returns already received during issue (overlap / starvation gauge)

NOTE: the phases overlap (returns stream back during issue), so the intervals are
NOT strictly additive; read them as composition, cross-checked with t1_payload.
"""

import os
import csv
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tracy.process_device_log import extract_device_info
from tests.tt_metal.tt_metal.data_movement.python.constants import DEFAULT_OUTPUT_DIR
from loguru import logger

PHASE_ZONES = {
    "dm_t0_issue_start": "t0",
    "dm_t1_issue_end": "t1",
    "dm_t2_first_return": "t2",
    "dm_t3_barrier_clear": "t3",
}
META_ZONES = {"Test id", "Number of transactions", "Transaction size in bytes"}


def parse_csv(file_path):
    """Return (meta_by_run, phase_by_run).

    meta_by_run[rhid][zone_name] = payload int
    phase_by_run[rhid][(x, y)][phase_key] = (timestamp, payload)
    """
    meta_by_run = {}
    phase_by_run = {}
    with open(file_path, "r") as f:
        f.readline()  # skip the "ARCH: ..." banner line
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            zone = row.get("zone name")
            if zone is None:
                continue
            if row.get("type") != "TS_DATA":
                continue
            try:
                rhid = int(row["run host ID"])
                payload = int(row["data"])
            except (TypeError, ValueError):
                continue

            if zone in META_ZONES:
                meta_by_run.setdefault(rhid, {})[zone] = payload
            elif zone in PHASE_ZONES:
                ts = int(row["time[cycles since reset]"])
                x = int(row["core_x"])
                y = int(row["core_y"])
                phase_by_run.setdefault(rhid, {}).setdefault((x, y), {})[PHASE_ZONES[zone]] = (ts, payload)
    return meta_by_run, phase_by_run


def select_run(meta_by_run, test_id, num_transactions, transaction_size):
    matches = [
        rhid
        for rhid, m in meta_by_run.items()
        if (test_id is None or m.get("Test id") == test_id)
        and (num_transactions is None or m.get("Number of transactions") == num_transactions)
        and (transaction_size is None or m.get("Transaction size in bytes") == transaction_size)
    ]
    if not matches:
        raise ValueError("No run matched the requested (test_id, num_transactions, transaction_size).")
    if len(matches) > 1:
        # Prefer the newest run (largest run host ID) rather than a dict-order pick.
        logger.warning(f"Multiple runs matched {matches}; using the newest (max run host ID).")
    return max(matches)


def compute_phase_maps(core_phases):
    """Return dict of {metric_name: {(x, y): value}} for one run's cores."""
    issue, first_byte, drain, total, t1_payload = {}, {}, {}, {}, {}
    for core, ph in core_phases.items():
        if not all(k in ph for k in ("t0", "t1", "t2", "t3")):
            logger.warning(f"Core {core} missing a phase marker; skipping.")
            continue
        t0 = ph["t0"][0]
        t1 = ph["t1"][0]
        t2 = ph["t2"][0]
        t3 = ph["t3"][0]
        issue[core] = t1 - t0
        first_byte[core] = t2 - t1
        drain[core] = t3 - t2
        total[core] = t3 - t0
        t1_payload[core] = ph["t1"][1]
    return {
        "issue (t1-t0)": issue,
        # t2 fires on the first *completed return* (whole page), not the first byte: a read
        # response, a non-posted write ack, or (posted writes) a request departing. The
        # progress counter increments once per transaction, so this is time-to-first-return.
        "first_return (t2-t1)": first_byte,
        "drain (t3-t2)": drain,
        "t1_payload (returns during issue)": t1_payload,
        "total (t3-t0)": total,
    }


def load_arch(file_path):
    arch, _, _ = extract_device_info(file_path)
    yaml_name = "blackhole_140_arch.yaml" if arch == "blackhole" else "wormhole_b0_80_arch.yaml"
    with open(os.path.join("tt_metal", "soc_descriptors", yaml_name), "r") as fh:
        arch_data = yaml.safe_load(fh)
    dram_x = sorted({int(c.split("-")[0]) for row in arch_data.get("dram", []) for c in row})
    return arch, arch_data, dram_x


def draw_panel(ax, fig, values_by_core, height, width, title, unit, dram_x):
    matrix = np.full((height, width), np.nan)
    for (x, y), v in values_by_core.items():
        matrix[y][x] = v
    worker = matrix[~np.isnan(matrix)]
    vmax = float(worker.max()) if worker.size else 1.0
    vmin = float(worker.min()) if worker.size else 0.0
    median = float(np.median(worker)) if worker.size else 0.0

    cmap = plt.get_cmap("hot").copy()
    cmap.set_bad("#dddddd")  # absent/arch tiles in grey, distinct from a true 0
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(vmax, 1))
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{title}")

    threshold = vmax / 2 if vmax > 0 else 1
    for y in range(height):
        for x in range(width):
            v = matrix[y][x]
            if np.isnan(v):
                continue
            color = "white" if v < threshold else "black"
            ax.text(x, y, f"{v:.0f}", ha="center", va="center", color=color, fontsize=6)

    # Mark DRAM columns so the spatial reasoning is anchored to the topology.
    for dx in dram_x:
        ax.axvline(dx, color="cyan", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(
        f"max={vmax:,.0f} min={vmin:,.0f} median={median:,.0f} {unit}",
        fontsize=8,
        family="monospace",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-core phase decomposition heatmaps (reader/writer)")
    parser.add_argument("-l", "--log_csv", default="generated/profiler/.logs/profile_log_device.csv")
    parser.add_argument("-i", "--test_id", type=int, required=True)
    parser.add_argument("-n", "--num_transactions", type=int, required=True)
    parser.add_argument("-s", "--transaction_size", type=int, required=True)
    parser.add_argument("-o", "--output_dir", default=None)
    args = parser.parse_args()

    meta_by_run, phase_by_run = parse_csv(args.log_csv)
    rhid = select_run(meta_by_run, args.test_id, args.num_transactions, args.transaction_size)
    logger.info(
        f"Selected run host ID {rhid} for test {args.test_id}, {args.num_transactions} tx, {args.transaction_size} B"
    )

    if rhid not in phase_by_run:
        raise ValueError(
            f"Run {rhid} has no phase markers. Rebuild/run with TT_DM_PHASE_COUNTERS=1 (and without NoC-event tracing)."
        )

    maps = compute_phase_maps(phase_by_run[rhid])
    arch, arch_data, dram_x = load_arch(args.log_csv)
    height = arch_data["grid"]["y_size"]
    width = arch_data["grid"]["x_size"]
    logger.info(f"Detected architecture: {arch}; DRAM columns at x={dram_x}")

    panels = [
        ("issue (t1-t0)", "cyc"),
        ("first_return (t2-t1)", "cyc"),
        ("drain (t3-t2)", "cyc"),
        ("total (t3-t0)", "cyc"),
        ("t1_payload (returns during issue)", "ret"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(30, 14))
    axes = axes.flatten()
    for ax, (name, unit) in zip(axes, panels):
        draw_panel(ax, fig, maps[name], height, width, name, unit, dram_x)
    axes[-1].axis("off")  # 6th cell unused (5 panels)

    fig.suptitle(
        f"Phase Decomposition ({arch.upper()}) - test {args.test_id}, "
        f"{args.num_transactions} tx x {args.transaction_size} B",
        fontsize=20,
        fontweight="bold",
    )

    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_DIR, arch, "multi_interleaved", "read_grid_sweep")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"phase_heatmap_{args.test_id}_0_{args.num_transactions}_{args.transaction_size}.png"
    )
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    for name, _unit in panels:
        vals = list(maps[name].values())
        if vals:
            print(f"{name:35s} min={min(vals):>10,.0f} median={np.median(vals):>10,.0f} max={max(vals):>10,.0f}")
    print(f"Saved phase heatmaps to {output_file}")
