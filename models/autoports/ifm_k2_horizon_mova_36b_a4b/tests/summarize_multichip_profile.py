# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only final report summaries, including per-rank EP and collective timing."""

import argparse
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/multichip_decoder"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, default=DOC)
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    doc, tag = args.artifact_dir, args.tag
    summaries = {}
    for path in sorted((doc / "tracy" / tag).glob("*/*_perf_report.csv")):
        rows = list(csv.DictReader(path.open()))
        categories = collections.defaultdict(float)
        for row in rows:
            categories[row["Op Category"]] += float(row["Device Time [μs]"])
        layer = 0 if path.parent.name == "dense" else 3
        audit = json.loads((doc / f"profile_{tag}_l{layer}_audit.json").read_text())
        assert audit["status"] == "pass"
        raw = list(csv.DictReader(open(audit["raw_csv"])))
        ranked = []
        for row in rows:
            if not any(op in row["OP Code"] for op in ("SparseMatmul", "ReduceScatter", "AllReduce")):
                continue
            group = int(row["Global Call Count"]) >> 10
            matches = [
                r
                for r in raw
                if r["DEVICE KERNEL DURATION [ns]"]
                and r["GLOBAL CALL COUNT"]
                and int(r["GLOBAL CALL COUNT"]) >> 10 == group
            ]
            ranked.append(
                dict(
                    op=row["OP Code"],
                    host_program=group,
                    executions=[
                        dict(
                            device=r["DEVICE ID"],
                            trace=r["METAL TRACE ID"],
                            replay=r["METAL TRACE REPLAY SESSION ID"],
                            kernel_us=float(r["DEVICE KERNEL DURATION [ns]"]) / 1000,
                        )
                        for r in matches
                    ],
                )
            )
        summaries[path.parent.name + "_" + path.name.split("_")[0]] = dict(
            category_us=dict(categories),
            sum_us=sum(categories.values()),
            raw_sha256=audit["raw_sha256"],
            top_ops=[
                {
                    k: r[k]
                    for k in (
                        "OP Code",
                        "Device Time [μs]",
                        "Cores",
                        "DRAM [%]",
                        "FLOPs [%]",
                        "Bound",
                        "Math Fidelity",
                        "Input 1 Datatype",
                        "Advice",
                    )
                }
                for r in sorted(rows, key=lambda r: float(r["Device Time [μs]"]), reverse=True)[:10]
            ],
            per_rank_sparse_and_ccl=ranked,
        )
    assert len(summaries) == 4
    (doc / "profile_findings.json").write_text(json.dumps(summaries, indent=2) + "\n")
    for key, value in summaries.items():
        print(key, value["category_us"])


if __name__ == "__main__":
    main()
