# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize bounded candidate jobs and retain exact commands and elapsed device-job time."""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--continue-on-failure", action="store_true")
    ap.add_argument("--layout", default="replicated")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    art = root / "doc/optimized_multichip_decoder"
    for name in args.candidates:
        for layer in map(int, args.layers.split(",")):
            if (art / "STOP").exists():
                return 75
            label = f"{name}_{args.layout}_{layer}"
            output = art / (label + ".json")
            if output.exists():
                continue
            cmd = [
                "timeout",
                "600",
                sys.executable,
                "-m",
                "models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe",
                "--layer",
                str(layer),
                "--layout",
                args.layout,
                "--optimization",
                name,
                "--lengths",
                "1024",
                "--trace",
                "--benchmark",
                "--reference",
                str(root / f"generated/multichip/perf_{layer}.pt"),
                "--output",
                str(output),
            ]
            row = dict(
                command=cmd,
                label=label,
                start=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                implementation_sha256=hashlib.sha256((root / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256((root / "tests/multichip_optimization.py").read_bytes()).hexdigest(),
                regime="serial",
                devices=[0, 1, 2, 3],
            )
            (art / (label + "_command.json")).write_text(json.dumps(row, indent=2) + "\n")
            print("START", label, flush=True)
            start = time.monotonic()
            with (art / (label + ".log")).open("w") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            row.update(
                seconds=time.monotonic() - start,
                exit_code=result.returncode,
                end=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            with (art / "device_jobs.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
            print("END", label, result.returncode, round(row["seconds"], 2), flush=True)
            if result.returncode:
                if not args.continue_on_failure:
                    return result.returncode
                recovery = subprocess.run(
                    ["bash", str(root / "tests/recover_stage5.sh"), label + "_recovery"],
                    env={**os.environ, "STAGE5_LOCK_HELD": "1"},
                )
                if recovery.returncode:
                    return recovery.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
