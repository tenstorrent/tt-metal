# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Pytest hooks for tt_dit/utils — currently just the sweep_mm_block_sizes summary."""

import csv
import os


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """After a sweep session, print a perf summary across all swept shapes.

    Reads sweep_results_mm.csv (written by test_mm_sweep), groups rows by
    (device_config, shape, use_case), and prints the best (lowest duration)
    config per group.
    """
    # Only activate if test_mm_sweep / test_mm_sweep_worker ran in this session.
    # Match on the function name segment (between "::" and "[" or end) so we
    # don't false-trigger on unrelated tests with "test_mm_sweep" in the name.
    sweep_fns = {"test_mm_sweep", "test_mm_sweep_worker"}

    def _is_sweep_node(nodeid):
        if not nodeid or "::" not in nodeid:
            return False
        fn = nodeid.rsplit("::", 1)[1].split("[", 1)[0]
        return fn in sweep_fns

    sweep_ran = any(
        _is_sweep_node(item.nodeid)
        for item in terminalreporter.stats.get("passed", []) + terminalreporter.stats.get("failed", [])
    )
    if not sweep_ran:
        return

    csv_path = "sweep_results_mm.csv"
    if not os.path.exists(csv_path):
        return

    # Group rows by (device_config, M, K, N, use_case); keep best OK row per group.
    best_by_group = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "OK":
                continue
            try:
                dur = float(row["device_kernel_duration_ns"])
            except ValueError:
                continue
            key = (row["device_config"], row["op_type"], row["use_case"], row["M"], row["K"], row["N"])
            cur = best_by_group.get(key)
            if cur is None or dur < cur["duration"]:
                best_by_group[key] = {
                    "duration": dur,
                    "M_block": row["M_block"],
                    "K_block": row["K_block"],
                    "N_block": row["N_block"],
                    "subblock_h": row["subblock_h"],
                    "subblock_w": row["subblock_w"],
                }

    if not best_by_group:
        return

    tr = terminalreporter
    tr.write_sep("=", "sweep perf summary")
    tr.write_line(
        f"{'device_config':<16} {'op':>5} {'use_case':>12} {'M':>6} {'K':>6} {'N':>6}"
        f" | {'M_blk':>5} {'K_blk':>5} {'N_blk':>5} {'sb_h':>4} {'sb_w':>4}"
        f" | {'duration_ns':>12}"
    )
    tr.write_line("-" * 116)
    for key, best in sorted(best_by_group.items()):
        device_config, op_type, use_case, M, K, N = key
        tr.write_line(
            f"{device_config:<16} {op_type:>5} {use_case:>12} {M:>6} {K:>6} {N:>6}"
            f" | {best['M_block']:>5} {best['K_block']:>5} {best['N_block']:>5}"
            f" {best['subblock_h']:>4} {best['subblock_w']:>4}"
            f" | {best['duration']:>12,.0f}"
        )
    tr.write_line(f"Full results: {csv_path}")
