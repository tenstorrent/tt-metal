# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bucket a tt_perf_report stacked CSV into the 3 presentation buckets.

Usage: bucket.py <config_id> <stage> <stacked_csv> [results_json]
Appends/updates one record in results_json (default ltx_tpsp_results.json).
"""
import csv
import json
import os
import sys

MATMUL_OPS = (
    "AllGatherMinimalMatmulAsyncOp",
    "MinimalMatmulStridedReduceScatterAsync",
    "MinimalMatmulDeviceOperation",
)
RING_OPS = ("RingJointSDPADeviceOperation",)

# TP/SP degrees per config id (mesh_shape + which axis is SP).
CONFIG_DEGREES = {
    "tp1_sp32": (1, 32), "tp2_sp16": (2, 16), "tp4_sp8": (4, 8),
    "tp8_sp4": (8, 4), "tp16_sp2": (16, 2), "tp32_sp1": (32, 1),
    "tp4_sp8_altaxis": (4, 8), "tp8_sp4_altaxis": (8, 4),
}


def bucket(stacked_csv):
    buckets = {"matmul_tp_ccl": 0.0, "ring_attention": 0.0, "overhead": 0.0}
    per_op = {}
    with open(stacked_csv) as f:
        for row in csv.DictReader(f):
            op = row["Op Code"].split(" (")[0].strip()
            us = float(row["Device Time Sum [μs]"])
            per_op[op] = per_op.get(op, 0.0) + us
            if op in MATMUL_OPS:
                buckets["matmul_tp_ccl"] += us
            elif op in RING_OPS:
                buckets["ring_attention"] += us
            else:
                buckets["overhead"] += us
    return buckets, per_op


def main():
    cid, stage, stacked_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    out = sys.argv[4] if len(sys.argv) > 4 else "ltx_tpsp_results.json"
    b, per_op = bucket(stacked_csv)
    tp, sp = CONFIG_DEGREES[cid]
    rec = {
        "config": cid, "stage": stage, "TP": tp, "SP": sp,
        "matmul_tp_ccl": round(b["matmul_tp_ccl"], 1),
        "ring_attention": round(b["ring_attention"], 1),
        "overhead": round(b["overhead"], 1),
        "total": round(sum(b.values()), 1),
        "per_op": {k: round(v, 1) for k, v in sorted(per_op.items(), key=lambda x: -x[1])},
    }
    data = {}
    if os.path.exists(out):
        with open(out) as f:
            data = json.load(f)
    data[f"{cid}/{stage}"] = rec
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps({k: rec[k] for k in ("config", "stage", "TP", "SP",
          "matmul_tp_ccl", "ring_attention", "overhead", "total")}, indent=2))


if __name__ == "__main__":
    main()
