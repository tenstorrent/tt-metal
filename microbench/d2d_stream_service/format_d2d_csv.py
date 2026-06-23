#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Post-process a Google-Benchmark CSV from benchmark_d2d_stream_service into a more
# readable form WITHOUT touching the benchmark binary or run_sweep.sh:
#   * prepend a "shape" column as column 1 (derived from size_index in the row name)
#   * sort rows from largest payload to smallest (payload_bytes descending)
#
# The size_index -> shape map MUST stay in sync with kThroughputShapes in
# benchmark_d2d_stream_service.cpp.
#
# Usage:
#   python3 microbench/d2d_stream_service/format_d2d_csv.py INPUT.csv [OUTPUT.csv]
#   # OUTPUT omitted -> writes to stdout
#   python3 .../format_d2d_csv.py d2d_bandwidth.csv d2d_bandwidth.sorted.csv

import csv
import re
import sys

# Keep in sync with kThroughputShapes (benchmark_d2d_stream_service.cpp).
SIZE_INDEX_TO_SHAPE = {
    0: "[1, 1, 256, 512]",  # 0.5 MB
    1: "[1, 1, 1024, 1024]",  # 4 MB
    2: "[1, 1, 4096, 1024]",  # 16 MB
    3: "[1, 1, 4096, 4096]",  # 64 MB
    4: "[1, 8, 4096, 4096]",  # 512 MB
}

_SIZE_INDEX_RE = re.compile(r"size_index:(\d+)")

# Columns dropped from the cleaned-up CSV:
#   * real_time / cpu_time                  -> Google-Benchmark wall/CPU time for the WHOLE
#                                              run_throughput call (service build + warmup +
#                                              capture + transfer), not the transfer itself.
#                                              Use transfer_ms / throughput_gbps instead.
#   * data_ok / error_*                     -> correctness / status columns.
#   * bytes_per_second / items_per_second   -> always empty for this benchmark.
DROP_COLUMNS = {
    "real_time",
    "cpu_time",
    "data_ok",
    "error_occurred",
    "error_message",
    "bytes_per_second",
    "items_per_second",
}


def shape_for_row(row):
    """Shape string for a row, or "" when the name carries no size_index (e.g. latency)."""
    m = _SIZE_INDEX_RE.search(row.get("name", ""))
    if m is None:
        return ""
    return SIZE_INDEX_TO_SHAPE.get(int(m.group(1)), "")


def payload_key(row):
    """Sort key: payload_bytes as float (missing/unparseable sorts last)."""
    try:
        return float(row.get("payload_bytes", ""))
    except (TypeError, ValueError):
        return float("-inf")


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if len(argv) >= 2 else 2

    in_path = argv[1]
    with open(in_path, newline="") as f:
        lines = f.readlines()
    # Google Benchmark prepends a context preamble (host info, "Load Average: ...") before
    # the CSV table; skip to the real header row, which starts with "name,".
    header_idx = next((i for i, ln in enumerate(lines) if ln.startswith("name,")), None)
    if header_idx is None:
        print(f"error: no 'name,...' header row found in {in_path}", file=sys.stderr)
        return 1
    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)
    fieldnames = list(reader.fieldnames)

    for row in rows:
        row["shape"] = shape_for_row(row)
    # Stable sort by payload descending; rows sharing a size keep their original order.
    rows.sort(key=payload_key, reverse=True)

    out_fields = ["shape"] + [f for f in fieldnames if f not in DROP_COLUMNS]
    out = open(argv[2], "w", newline="") if len(argv) > 2 else sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
