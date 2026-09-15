# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-operator mode-4 block sweep; each candidate runs in a fresh process."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REPRO = ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py"
BLOCKS = [(q, k) for q in (32, 64, 128, 256) for k in (256, 512, 1024)]
BLOCKS += [(32, 2048), (64, 2048), (128, 2048), (256, 128)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "long", "final", "stress"), required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32768])
    args = parser.parse_args()
    env = dict(os.environ, TT_SDPA_ACCURACY_DIAG="4", TT_SDPA_BLOCK_SWEEP="1")
    source_hashes = {}
    for line in (OUT / "SOURCE-SHA256.txt").read_text().splitlines():
        expected, path = line.split(maxsplit=1)
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert actual == expected, path
        source_hashes[path] = actual
    screen = []
    for path in OUT.glob("screen-n32768-*.jsonl"):
        screen.extend(json.loads(x) for x in path.read_text().splitlines())
    screen = sorted((r for r in screen if r["l2_pct"] <= 0.25), key=lambda r: r["trace_median_ms"])
    finalists = list(dict.fromkeys([(r["q_chunk"], r["k_chunk"]) for r in screen] + [(128, 1024)]))
    for n in args.lengths:
        if args.stage == "screen":
            blocks = BLOCKS
        elif args.stage == "long":
            assert screen, "Run the 32K screen first"
            blocks = finalists
        else:
            candidates = []
            stage = "screen" if n == 32768 else "long"
            for path in OUT.glob(f"{stage}-n{n}-*.jsonl"):
                candidates.extend(json.loads(x) for x in path.read_text().splitlines())
            candidates = sorted((r for r in candidates if r["l2_pct"] <= 0.25), key=lambda r: r["trace_median_ms"])
            assert candidates, f"No eligible candidates at N={n}"
            count = 2 if args.stage == "final" else 1
            blocks = list(dict.fromkeys([(r["q_chunk"], r["k_chunk"]) for r in candidates[:count]] + [(128, 1024)]))
        distributions = ("scaled_qk", "outliers") if args.stage == "stress" else ("normal",)
        for q, k in blocks:
            for dist in distributions:
                label = f"{args.stage}-n{n}-q{q}-k{k}-{dist}"
                result = OUT / f"{label}.jsonl"
                status = OUT / f"{label}.status.json"
                if status.exists():
                    print("SKIP " + label, flush=True)
                    continue
                warmup, iters = (40, 10) if args.stage == "final" else (10, 5)
                if args.stage == "stress":
                    warmup, iters = 0, 1
                rows = 512 if args.stage in ("final", "stress") else 128
                cmd = [
                    sys.executable,
                    str(REPRO),
                    "--kv-lens",
                    str(n),
                    "--full",
                    "--heads",
                    "10",
                    "--q-chunk",
                    str(q),
                    "--k-chunks",
                    str(k),
                    "--variants",
                    "fp32_hifi2",
                    "--seed",
                    "1236",
                    "--q-len",
                    str(rows),
                    "--query-sampling",
                    "spread",
                    "--benchmark-warmup",
                    str(warmup),
                    "--benchmark-iters",
                    str(iters),
                    "--distribution",
                    dist,
                    "--per-head-metrics",
                    "--label",
                    label,
                    "--output",
                    str(result),
                ]
                if args.stage in ("final", "stress"):
                    cmd.append("--check-full-output")
                print("START " + label, flush=True)
                start = time.monotonic()
                with (OUT / f"{label}.log").open("w") as log:
                    run = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
                record = dict(
                    label=label,
                    command=cmd,
                    returncode=run.returncode,
                    elapsed_s=time.monotonic() - start,
                    source_sha256=source_hashes,
                )
                if result.exists() and result.stat().st_size:
                    r = json.loads(result.read_text().splitlines()[-1])
                    record.update(
                        trace_median_ms=r["trace_median_ms"],
                        l2_pct=r["l2_pct"],
                        useful_tflops=4 * 10 * n * n * 128 / (r["trace_median_ms"] * 1e9),
                    )
                status.write_text(json.dumps(record, indent=2) + "\n")
                print("RESULT " + json.dumps(record), flush=True)
                if run.returncode:
                    log_text = (OUT / f"{label}.log").read_text()
                    capacity_rejection = "L1" in log_text and any(
                        s in log_text for s in ("allocate", "allocation", "out of memory")
                    )
                    clean_trace_failure = (
                        "Tensor-likes are not equal!" in log_text and "Cluster destructor completed" in log_text
                    )
                    if not (capacity_rejection or clean_trace_failure):
                        raise RuntimeError(f"Unexpected failure; inspect {label}.log before continuing")


if __name__ == "__main__":
    main()
