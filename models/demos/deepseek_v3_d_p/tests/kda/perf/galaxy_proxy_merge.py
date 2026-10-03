# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Merge the ``test_galaxy_sp`` and ``test_galaxy_tp`` Tracy profiles into Galaxy's KDA op list.

Per op: slowest chip's device kernel time in the pass after the ``galaxy_proxy`` signpost.
Galaxy's op list = SP-part ops, then the TP reduce-scatter.

    python galaxy_proxy_merge.py SP_ops_perf_results.csv TP_ops_perf_results.csv [-o merged.csv]
"""

from __future__ import annotations

import argparse

import pandas as pd

KERNEL_NS = "DEVICE KERNEL DURATION [ns]"


def measured_ops(csv_path: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, low_memory=False)
    signpost = frame.index[(frame["OP TYPE"] == "signpost") & (frame["OP CODE"] == "galaxy_proxy")][-1]
    frame = frame.loc[signpost + 1 :]
    frame = frame[frame["OP TYPE"] == "tt_dnn_device"].copy()
    frame["op_index"] = frame.groupby("DEVICE ID").cumcount()
    ops = frame.groupby("op_index").agg(op=("OP CODE", "first"), us=(KERNEL_NS, lambda ns: ns.max() / 1e3))
    ops["op"] = ops["op"].str.removesuffix("DeviceOperation").str.removesuffix("Operation")
    return ops.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sp_csv")
    parser.add_argument("tp_csv")
    parser.add_argument("-o", "--output", help="write the merged per-op table as CSV")
    args = parser.parse_args()

    merged = pd.concat([measured_ops(args.sp_csv), measured_ops(args.tp_csv)], ignore_index=True)
    if args.output:
        merged.to_csv(args.output, index_label="op_index")
    by_op = merged.groupby("op")["us"].agg(["count", "sum"]).sort_values("sum", ascending=False)
    by_op["pct"] = 100.0 * by_op["sum"] / merged["us"].sum()
    print(by_op.to_string(float_format=lambda value: f"{value:8.1f}"))
    print(f"\n{len(merged)} ops, kernel sum {merged['us'].sum() / 1e3:.3f} ms")


if __name__ == "__main__":
    main()
