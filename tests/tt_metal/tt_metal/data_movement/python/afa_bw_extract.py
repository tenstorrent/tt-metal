#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract delivered aggregate bandwidth for all_from_all (read) tests 310/311/319.

Reads run on RISCV_1 (requestor kernel). Uses StatsCollector.combined_bandwidth.
"""
import sys
import csv

from tests.tt_metal.tt_metal.data_movement.python.stats_collector import StatsCollector

NOC_GHZ = 1.35  # 1 B/cyc = 1.35 GB/s


def main():
    log = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else None
    risc = "riscv_1"  # reads

    collector = StatsCollector(log, test_id_to_name={}, test_type_attributes={}, verbose=False)
    _, agg = collector.gather_analysis_stats()
    if risc not in agg or not agg[risc]:
        raise SystemExit(f"No data for {risc}")

    rows = []
    for run_host_id, data in agg[risc].items():
        attrs = data["attributes"]
        tid = attrs.get("Test id")
        N = attrs.get("Transaction size in bytes", 0)
        Q = attrs.get("Number of transactions", 0)
        cores = data["num_cores"]
        wall = data["wall_clock_time"]
        bw = data["combined_bandwidth"]
        rows.append(
            {
                "run_host_id": run_host_id,
                "test_id": tid,
                "cores": cores,
                "txn_size_B": N,
                "num_txn_per_core": Q,
                "wall_cyc": round(wall, 1),
                "agg_bw_bpc": round(bw, 2),
                "agg_bw_gbps": round(bw * NOC_GHZ, 1),
                "per_core_bpc": round(bw / cores, 3) if cores else 0,
            }
        )

    rows.sort(key=lambda r: (r["test_id"] or 0, r["txn_size_B"], r["num_txn_per_core"]))

    hdr = [
        "run_host_id",
        "test_id",
        "cores",
        "txn_size_B",
        "num_txn_per_core",
        "wall_cyc",
        "agg_bw_bpc",
        "agg_bw_gbps",
        "per_core_bpc",
    ]
    print("\t".join(hdr))
    for r in rows:
        print("\t".join(str(r[h]) for h in hdr))

    # summary: peak per test_id
    by_tid = {}
    for r in rows:
        t = r["test_id"]
        if t not in by_tid or r["agg_bw_bpc"] > by_tid[t]["agg_bw_bpc"]:
            by_tid[t] = r
    print("\n=== PEAK delivered aggregate BW per test_id ===")
    for t, r in sorted(by_tid.items(), key=lambda x: x[0] or 0):
        print(
            f"test {t}: peak {r['agg_bw_bpc']} B/cyc ({r['agg_bw_gbps']} GB/s) @ N={r['txn_size_B']}B, Q={r['num_txn_per_core']}, per-core {r['per_core_bpc']} B/cyc, cores={r['cores']}"
        )

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
