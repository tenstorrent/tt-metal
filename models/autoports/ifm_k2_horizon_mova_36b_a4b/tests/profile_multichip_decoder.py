# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized final Tracy capture, integrity audit, and tt-perf-report artifacts."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/multichip_decoder"
MODULE = "models.autoports.ifm_k2_horizon_mova_36b_a4b.tests."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", type=int, choices=[0, 3], default=[0, 3])
    ap.add_argument("--artifact-dir", type=Path, default=DOC)
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    doc = args.artifact_dir
    for layer in args.layers:
        out = doc / "local" / f"profile_{args.tag}_l{layer}"
        log = doc / "logs" / f"profile_{args.tag}_l{layer}.log"
        cmd = [
            sys.executable,
            "-m",
            "tracy",
            "-r",
            "-p",
            "-v",
            "--no-web-server",
            "-o",
            str(out),
            "-m",
            MODULE + "run_multichip_decoder",
            "--layer",
            str(layer),
            "--seq",
            "2048",
            "--context",
            "4096",
            "--trace",
            "--profile",
            "--iterations",
            "1",
            "--tag",
            f"profile_{args.tag}",
            "--artifact-dir",
            str(doc),
        ]
        env = {**os.environ, "TT_METAL_TRACE_ALLOC_TRACKING": "0", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0"}
        for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_LOGS_PATH"):
            env.pop(key, None)
        print("START PROFILE", layer, flush=True)
        start = time.time()
        with log.open("w") as f:
            process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                print("TRIAGE_REQUIRED left alive", process.pid, flush=True)
                raise
        record = dict(
            command=cmd,
            exit_code=code,
            seconds=time.time() - start,
            source_sha256=hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
            log=str(log),
        )
        (doc / f"profile_{args.tag}_l{layer}_execution.json").write_text(json.dumps(record, indent=2) + "\n")
        assert code == 0, record
        raw = max(out.glob("reports/*/ops_perf_results_*.csv"), key=lambda p: p.stat().st_mtime)
        subprocess.run(
            [
                sys.executable,
                "-m",
                MODULE + "audit_multichip_profile",
                str(raw),
                str(log),
                str(doc / f"profile_{args.tag}_l{layer}_audit.json"),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                MODULE + "multichip_perf_report",
                str(layer),
                str(raw),
                "--tag",
                args.tag,
                "--artifact-dir",
                str(doc),
            ],
            check=True,
        )
        print("DONE PROFILE", layer, flush=True)


if __name__ == "__main__":
    main()
