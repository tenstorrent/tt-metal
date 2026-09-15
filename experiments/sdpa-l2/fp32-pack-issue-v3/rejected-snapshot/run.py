# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fresh-process scheduling controls with complete source and output provenance."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCES = [
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_fp32_pipeline.hpp",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp",
    "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_fp32_l1_pipeline.hpp",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=["control", "fixed", "l1", "l1macro", "l1pipe", "light", "refine"], required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32768, 131072])
    parser.add_argument("--distributions", nargs="+", default=["normal"])
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1236)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--repeat-pack", action="store_true")
    args = parser.parse_args()
    env = {k: v for k, v in os.environ.items() if not k.startswith("TT_SDPA_")}
    env.update(
        {
            "TT_SDPA_ACCURACY_DIAG": "4",
            "TT_SDPA_FP32_SUB_BATCH": "2",
            "TT_SDPA_FP32_FUSED_EXP": "1",
            "TT_SDPA_FP32_REUSE_EXP": "1",
            "TT_SDPA_FP32_EXTRA_CONST": "1",
            "TT_SDPA_FP32_PAIRED_UNPACK": "1",
            "TT_SDPA_FP32_PAIRED_PACK": "1",
            "TT_SDPA_DENOM_PHASES": "2",
        }
    )
    if args.mode == "fixed":
        env["TT_SDPA_FP32_PIPELINE"] = "1"
    if args.mode in ("l1", "l1macro", "l1pipe", "light", "refine"):
        env["TT_SDPA_FP32_L1_SUB"] = "1"
    if args.mode in ("l1macro", "l1pipe", "light", "refine"):
        env["TT_SDPA_FP32_L1_MACRO"] = "1"
    if args.mode == "l1pipe":
        env["TT_SDPA_FP32_L1_PIPELINE"] = "1"
    if args.mode == "light":
        env["TT_SDPA_FP32_LIGHT_RELOAD"] = "1"
    if args.mode == "refine":
        env["TT_SDPA_FP32_REFINE_MACRO"] = "1"
    if args.repeat_pack:
        assert args.mode in ("l1", "l1macro")
        env["TT_SDPA_FP32_L1_REPEAT"] = "1"
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}
    for n in args.lengths:
        for dist in args.distributions:
            label = f"{args.label}-n{n}-{dist}"
            result = HERE / (label + ".jsonl")
            assert not result.exists(), "Use a fresh label"
            cmd = [
                sys.executable,
                "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py",
                "--kv-lens",
                str(n),
                "--full",
                "--heads",
                str(args.heads),
                "--q-chunk",
                "128",
                "--k-chunks",
                "1024",
                "--variants",
                "fp32_hifi2",
                "--seed",
                str(args.seed),
                "--q-len",
                "512",
                "--query-sampling",
                "spread",
                "--distribution",
                dist,
                "--benchmark-warmup",
                str(args.warmup),
                "--benchmark-iters",
                str(args.iters),
                "--per-head-metrics",
                "--check-full-output",
                "--label",
                label,
                "--output",
                str(result),
            ]
            (HERE / (label + ".provenance.json")).write_text(
                json.dumps(
                    dict(
                        source_sha256=hashes,
                        command=cmd,
                        environment={k: v for k, v in env.items() if k.startswith("TT_SDPA_")},
                    ),
                    indent=2,
                )
                + "\n"
            )
            print("START " + label, flush=True)
            with (HERE / (label + ".log")).open("w") as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
            r = json.loads(result.read_text())
            assert r["full_trace_equality"] and r["full_output_finite_checked"]
            print(
                "RESULT "
                + json.dumps({k: r[k] for k in ("label", "trace_median_ms", "l2_pct", "pcc", "full_output_sha256")}),
                flush=True,
            )


if __name__ == "__main__":
    main()
