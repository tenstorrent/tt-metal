# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Paired baseline/candidate sustained timings and exact-output stress checks."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pair(prefix, n, distributions, qk_width, heads=10, seed=1236, warmup=0, iters=1):
    for candidate in (False, True):
        label = prefix + ("-candidate" if candidate else "-baseline")
        cmd = [
            sys.executable,
            str(HERE / "run.py"),
            "--label",
            label,
            "--lengths",
            str(n),
            "--heads",
            str(heads),
            "--seed",
            str(seed),
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--batches",
            "2" if candidate else "1",
            "--distributions",
            *distributions,
        ]
        if candidate:
            cmd += [
                "--fused",
                "--reuse-exp",
                "--extra-const",
                "--paired-unpack",
                "--paired-pack",
                "--denom-phases",
                "2",
                "--qk-width",
                str(qk_width),
            ]
        subprocess.run(cmd, check=True)
    for dist in distributions:
        a = json.loads((HERE / f"{prefix}-baseline-n{n}-{dist}-b1.jsonl").read_text())
        b = json.loads((HERE / f"{prefix}-candidate-n{n}-{dist}-b2.jsonl").read_text())
        assert a["full_output_sha256"] == b["full_output_sha256"], (prefix, n, dist, "full-output mismatch")
        assert a["l2_pct"] == b["l2_pct"] and a["pcc"] == b["pcc"]
        assert a["full_trace_equality"] and b["full_trace_equality"]
        print(
            "VALIDATED "
            + json.dumps(
                dict(
                    prefix=prefix,
                    n=n,
                    distribution=dist,
                    heads=heads,
                    seed=seed,
                    baseline_ms=a["trace_median_ms"],
                    candidate_ms=b["trace_median_ms"],
                    l2_pct=b["l2_pct"],
                    pcc=b["pcc"],
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--qk-width", type=int, choices=[2, 4], required=True)
    parser.add_argument("--section", choices=["perf", "stress", "holdout", "all"], default="all")
    args = parser.parse_args()
    if args.section in ("perf", "all"):
        for n in (32768, 131072, 262144):
            pair(args.label + "-perf", n, ["normal"], args.qk_width, warmup=40, iters=10)
    if args.section in ("stress", "all"):
        pair(
            args.label + "-stress",
            32768,
            [
                "normal",
                "scaled_qk",
                "outliers",
                "uniform",
                "constant_v",
                "uniform_constant_v",
                "biased_v",
                "common_q",
                "common_k",
                "common_v",
            ],
            args.qk_width,
        )
    if args.section in ("holdout", "all"):
        pair(args.label + "-holdout", 65536, ["normal", "scaled_qk", "outliers"], args.qk_width, heads=5, seed=1237)
