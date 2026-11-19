# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize bounded precision jobs; a stop file stops only between device jobs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--suffix", default="v1")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    root = Path(__file__).resolve().parents[1]
    evidence = root / "doc/datatype_sweep"
    prefix = "models.autoports.qwen_qwen3_8_flash_next.tests."
    for cid in args.configs:
        if (evidence / "STOP_QUEUE").exists():
            print("STOP_QUEUE observed between jobs", flush=True)
            return
        stem = f"{cid}_{args.suffix}"
        output = evidence / f"{stem}.json"
        if output.exists():
            raise FileExistsError(f"Preserve existing evidence: {output}")
        command = [
            sys.executable,
            "-m",
            prefix + "run_recorded",
            "--record",
            str(evidence / f"{stem}.provenance.json"),
            "--log",
            str(evidence / "logs" / f"{stem}.log"),
            "--",
            "timeout",
            "5400",
            sys.executable,
            "-m",
            prefix + "run_datatype_candidate",
            "--build-kwargs-file",
            str(evidence / "configs" / f"{cid}.kwargs.json"),
            "--reference",
            str(root / "readiness_aime24_chat.refpt"),
            "--reference",
            str(evidence / "natural_text.refpt"),
            "--output",
            str(output),
        ]
        repeats = max(args.repeats, 2 if cid == "inherited_e4_lofi" else 1)
        if repeats > 1:
            command.extend(["--repeats", str(repeats)])
        print(json.dumps({"starting": cid, "command": command}), flush=True)
        result = subprocess.run(command)
        if result.returncode:
            raise SystemExit(result.returncode)
        report = json.loads(output.read_text())
        print(json.dumps({"finished": cid, "passed": report["passed"], "completed": report["completed"]}), flush=True)


if __name__ == "__main__":
    main()
