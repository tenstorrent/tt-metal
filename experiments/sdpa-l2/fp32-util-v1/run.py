# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run isolated FP32 batching controls with full-output fingerprints."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = [
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--lengths", type=int, nargs="+", default=[32768])
    parser.add_argument("--distributions", nargs="+", default=["normal"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--cache-max", action="store_true")
    parser.add_argument("--reuse-exp", action="store_true")
    parser.add_argument("--extra-const", action="store_true")
    parser.add_argument("--shadow-max", action="store_true")
    parser.add_argument("--paired-unpack", action="store_true")
    parser.add_argument("--paired-pack", action="store_true")
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1236)
    parser.add_argument("--qk-width", type=int, choices=[2, 4], default=4)
    parser.add_argument("--qk-height", type=int, choices=[1, 2], default=1)
    parser.add_argument("--denom-phases", type=int, choices=[2, 3, 4], default=4)
    args = parser.parse_args()
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}
    for n in args.lengths:
        for dist in args.distributions:
            outputs = []
            for batch in args.batches:
                label = f"{args.label}-n{n}-{dist}-b{batch}"
                result = OUT / f"{label}.jsonl"
                assert not result.exists(), f"Use a fresh label: {result}"
                env = dict(os.environ, TT_SDPA_ACCURACY_DIAG="4", TT_SDPA_FP32_SUB_BATCH=str(batch))
                env.pop("TT_SDPA_FP32_FUSED_EXP", None)
                if args.fused:
                    env["TT_SDPA_FP32_FUSED_EXP"] = "1"
                env.pop("TT_SDPA_FP32_CACHE_MAX", None)
                if args.cache_max:
                    env["TT_SDPA_FP32_CACHE_MAX"] = "1"
                env.pop("TT_SDPA_FP32_REUSE_EXP", None)
                if args.reuse_exp:
                    env["TT_SDPA_FP32_REUSE_EXP"] = "1"
                env.pop("TT_SDPA_FP32_EXTRA_CONST", None)
                if args.extra_const:
                    env["TT_SDPA_FP32_EXTRA_CONST"] = "1"
                env["TT_SDPA_DENOM_PHASES"] = str(args.denom_phases)
                env.pop("TT_SDPA_FP32_SHADOW_MAX", None)
                if args.shadow_max:
                    env["TT_SDPA_FP32_SHADOW_MAX"] = "1"
                env.pop("TT_SDPA_FP32_PAIRED_UNPACK", None)
                if args.paired_unpack:
                    env["TT_SDPA_FP32_PAIRED_UNPACK"] = "1"
                env.pop("TT_SDPA_FP32_PAIRED_PACK", None)
                if args.paired_pack:
                    env["TT_SDPA_FP32_PAIRED_PACK"] = "1"
                env.pop("TT_SDPA_FP32_QK_WIDTH", None)
                if args.qk_width != 4:
                    env["TT_SDPA_FP32_QK_WIDTH"] = str(args.qk_width)
                env.pop("TT_SDPA_FP32_QK_HEIGHT", None)
                if args.qk_height != 1:
                    env["TT_SDPA_FP32_QK_HEIGHT"] = str(args.qk_height)
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
                    str(args.rows),
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
                (OUT / f"{label}.provenance.json").write_text(
                    json.dumps(dict(command=cmd, source_sha256=hashes), indent=2) + "\n"
                )
                print("START " + label, flush=True)
                with (OUT / f"{label}.log").open("w") as log:
                    subprocess.run(
                        cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200
                    )
                r = json.loads(result.read_text())
                outputs.append(r)
                print(
                    "RESULT "
                    + json.dumps({k: r[k] for k in ("label", "trace_median_ms", "l2_pct", "full_output_sha256")}),
                    flush=True,
                )
            if len(outputs) > 1:
                assert len({r["full_output_sha256"] for r in outputs}) == 1, "Batched output differs from batch 1"
                print("EXACT_MATCH " + str(n) + " " + dist, flush=True)


if __name__ == "__main__":
    main()
