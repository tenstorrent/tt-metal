# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate the retained L1-subtraction/macro candidate against frozen outputs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "fp32-util-v1"


def run(label, mode, n, distributions, heads=10, seed=1236, warmup=0, iters=1):
    subprocess.run(
        [
            sys.executable,
            str(HERE / "run.py"),
            "--label",
            label,
            "--mode",
            mode,
            "--lengths",
            str(n),
            "--distributions",
            *distributions,
            "--heads",
            str(heads),
            "--seed",
            str(seed),
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
        ],
        check=True,
    )


def check(a, b):
    assert a["full_output_sha256"] == b["full_output_sha256"], (a["label"], b["label"], "full output mismatch")
    assert a["l2_pct"] == b["l2_pct"] and a["pcc"] == b["pcc"]
    assert a["full_trace_equality"] and b["full_trace_equality"]
    print(
        "VALIDATED "
        + json.dumps(
            dict(
                label=b["label"],
                baseline_ms=a["trace_median_ms"],
                candidate_ms=b["trace_median_ms"],
                l2_pct=b["l2_pct"],
                pcc=b["pcc"],
                full_output_sha256=b["full_output_sha256"],
            )
        ),
        flush=True,
    )


def read(path):
    return json.loads(path.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    for n in (32768, 131072, 262144):
        labels = [args.label + "-perf-" + mode for mode in ("control", "l1macro")]
        for label, mode in zip(labels, ("control", "l1macro")):
            run(label, mode, n, ["normal"], warmup=40, iters=10)
        a, b = [read(HERE / f"{label}-n{n}-normal.jsonl") for label in labels]
        check(a, b)
        frozen = read(FROZEN / f"final-v1-perf-baseline-n{n}-normal-b1.jsonl")
        assert frozen["full_output_sha256"] == b["full_output_sha256"]
    for section, n, heads, seed, distributions in [
        (
            "stress",
            32768,
            10,
            1236,
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
        ),
        ("holdout", 65536, 5, 1237, ["normal", "scaled_qk", "outliers"]),
    ]:
        label = args.label + "-" + section
        run(label, "l1macro", n, distributions, heads=heads, seed=seed)
        for dist in distributions:
            a = read(FROZEN / f"final-v1-{section}-baseline-n{n}-{dist}-b1.jsonl")
            b = read(HERE / f"{label}-n{n}-{dist}.jsonl")
            check(a, b)
