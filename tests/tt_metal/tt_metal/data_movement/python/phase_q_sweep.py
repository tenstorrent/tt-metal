# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Q-sweep of the read-phase decomposition to locate the outstanding-transaction window.

For a fixed transaction (page) size, aggregate the t0..t3 phase markers across every
Q = "Number of transactions" present in the log and report, per Q:

    issue  (median t1-t0)      - backpressured injection time
    drain  (median t3-t2)      - final in-flight window drain
    total  (median t3-t0)
    outstanding_at_issue_end   = Q - t1_payload, per core; report median and max

Interpretation:
  * For Q <= W (window): issue ~ flat/small, drain dominates, outstanding tracks Q 1:1.
  * For Q  > W: issue grows ~linearly with Q (backpressure), outstanding plateaus at ~W.
  The knee in total-vs-Q, the issue/drain crossover, and the outstanding plateau all
  estimate the hardware outstanding-read window W.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from tests.tt_metal.tt_metal.data_movement.python.phase_heatmap import parse_csv, PHASE_ZONES


def aggregate_by_q(meta_by_run, phase_by_run, test_id, transaction_size):
    """Return sorted list of dicts, one per Q, with median phase metrics."""
    # Group runs by Q for the requested (test_id, transaction_size).
    runs_by_q = {}
    for rhid, m in meta_by_run.items():
        if test_id is not None and m.get("Test id") != test_id:
            continue
        if transaction_size is not None and m.get("Transaction size in bytes") != transaction_size:
            continue
        q = m.get("Number of transactions")
        if q is None:
            continue
        # Newest run wins if a (Q, size) repeats.
        if q not in runs_by_q or rhid > runs_by_q[q]:
            runs_by_q[q] = rhid

    rows = []
    for q in sorted(runs_by_q):
        rhid = runs_by_q[q]
        cores = phase_by_run.get(rhid, {})
        issue, drain, total, outstanding = [], [], [], []
        t0_abs, t3_abs = [], []
        for ph in cores.values():
            if not all(k in ph for k in ("t0", "t1", "t2", "t3")):
                continue
            t0, t1, t2, t3 = ph["t0"][0], ph["t1"][0], ph["t2"][0], ph["t3"][0]
            issue.append(t1 - t0)
            drain.append(t3 - t2)
            total.append(t3 - t0)
            outstanding.append(q - ph["t1"][1])  # Q - responses received during issue
            t0_abs.append(t0)
            t3_abs.append(t3)
        if not total:
            continue
        # Wall time is set by the straggler; decompose THAT core so the phase
        # latencies add up to the wall-bound total (issue + first_byte + drain).
        tail = int(np.argmax(total))
        rows.append(
            {
                "Q": q,
                # tail-core (wall-defining) phase latencies
                "issue_wall": float(issue[tail]),
                "drain_wall": float(drain[tail]),
                "total_max": float(total[tail]),
                # true wall span across cores (needs chip-wide synced clock).
                "wall_span": float(max(t3_abs) - min(t0_abs)),
                "outstanding_median": float(np.median(outstanding)),
                "outstanding_max": float(np.max(outstanding)),
                "n_cores": len(total),
            }
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-sweep phase decomposition / outstanding-window estimator")
    parser.add_argument("-l", "--log_csv", default="generated/profiler/.logs/profile_log_device.csv")
    parser.add_argument("-i", "--test_id", type=int, required=True)
    parser.add_argument("-s", "--transaction_size", type=int, required=True)
    parser.add_argument("-o", "--output", default="generated/profiler/phase_q_sweep.png")
    args = parser.parse_args()

    meta_by_run, phase_by_run = parse_csv(args.log_csv)
    rows = aggregate_by_q(meta_by_run, phase_by_run, args.test_id, args.transaction_size)
    if not rows:
        raise ValueError("No phase-marked runs found for the requested test_id/transaction_size.")

    header = (
        f"{'Q':>6} {'issue_wall':>12} {'drain_wall':>12} {'total_max':>12} {'wall':>12} "
        f"{'out_med':>10} {'out_max':>10} {'cores':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['Q']:>6} {r['issue_wall']:>12,.0f} {r['drain_wall']:>12,.0f} {r['total_max']:>12,.0f} "
            f"{r['wall_span']:>12,.0f} {r['outstanding_median']:>10,.1f} {r['outstanding_max']:>10,.0f} "
            f"{r['n_cores']:>6}"
        )

    qs = [r["Q"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(qs, [r["total_max"] for r in rows], "o-", color="crimson", label="total latency (wall / tail core)")
    ax1.plot(qs, [r["issue_wall"] for r in rows], "s-", color="gray", label="issue latency (tail core, t1-t0)")
    ax1.plot(qs, [r["drain_wall"] for r in rows], "^-", color="green", label="drain latency (tail core, t3-t2)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Q = number of transactions")
    ax1.set_ylabel("cycles (wall-defining core)")
    ax1.set_title(f"Wall-time phase decomposition vs Q (test {args.test_id}, {args.transaction_size} B)")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(qs, [r["outstanding_median"] for r in rows], "o-", label="outstanding @ issue-end (median)")
    ax2.plot(qs, [r["outstanding_max"] for r in rows], "s-", label="outstanding @ issue-end (max)")
    ax2.plot(qs, qs, "k--", alpha=0.5, label="y = Q (1:1, no backpressure)")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Q = number of transactions")
    ax2.set_ylabel("outstanding reads at issue-end")
    ax2.set_title("Outstanding window: departs y=Q and plateaus at ~W")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to {args.output}")
