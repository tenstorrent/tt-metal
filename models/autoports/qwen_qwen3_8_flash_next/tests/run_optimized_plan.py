# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run stage jobs serially; a STOP marker takes effect between jobs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    progress = dict(status="running", rows=[])
    output = a.plan.with_name(a.plan.stem + "_progress.json")
    for job in plan["jobs"]:
        if a.plan.with_name("STOP_" + a.plan.stem).exists():
            progress["status"] = "stopped_between_jobs"
            break
        command = [
            sys.executable,
            "-m",
            "models.autoports.qwen_qwen3_8_flash_next.tests.run_optimized_job",
            "--name",
            job["name"],
            "--seconds",
            str(job.get("seconds", 1800)),
        ]
        if job.get("tracked"):
            command.append("--tracked")
        if job.get("watcher"):
            command.append("--watcher")
        result = subprocess.run([*command, "--", *job["argv"]], env=None)
        progress["rows"].append(dict(name=job["name"], exit_code=result.returncode))
        output.write_text(json.dumps(progress, indent=2) + "\n")
        if result.returncode in (124, 137) or (result.returncode and not job.get("continue_on_failure", False)):
            progress["status"] = "failed"
            break
    else:
        progress["status"] = (
            "completed_with_failures" if any(row["exit_code"] for row in progress["rows"]) else "completed"
        )
    output.write_text(json.dumps(progress, indent=2) + "\n")
    if progress["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
