# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize an explicit stage job plan; stop between jobs, never mid-dispatch."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    for job in json.loads(args.plan.read_text()):
        if args.stop_file and args.stop_file.exists():
            print("STOPPED_BETWEEN_JOBS", flush=True)
            return
        environment = os.environ.copy()
        for key, value in job.get("environment", {}).items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        command = [
            sys.executable,
            "-m",
            "models.autoports.qwen_qwen3_8_flash_next.tests.run_recorded",
            "--record",
            job["record"],
            "--log",
            job["log"],
            "--",
            *job["argv"],
        ]
        result = subprocess.run(command, env=environment)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
