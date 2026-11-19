# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial bounded fused-stage jobs with a frozen runtime and between-job stop file."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    model = root.parent.parent
    source = model / "tt/fused_decoder.py"
    progress = dict(
        plan=str(args.plan), plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(), rows=[], status="running"
    )
    progress_path = root / (args.plan.stem + "_progress.json")

    def save():
        progress["updated_utc"] = datetime.now(timezone.utc).isoformat()
        progress_path.write_text(json.dumps(progress, indent=2) + "\n")

    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith(("TT_METAL_PROFILER", "TT_METAL_STREAMING_PROFILER")) or key in (
            "TT_METAL_DEVICE_PROFILER",
            "TT_METAL_TRACE_PROFILER",
            "TT_METAL_WATCHER",
            "TT_METAL_LOGS_PATH",
        ):
            del environment[key]
    environment.update(
        TT_VISIBLE_DEVICES="0",
        TT_METAL_TRACE_ALLOC_TRACKING="1",
        TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
        TT_METAL_TRACE_ALLOC_TRACEBACKS="0",
    )
    try:
        for job in plan["jobs"]:
            if (root / ("STOP_" + args.plan.stem)).exists():
                progress["status"] = "stopped_between_jobs"
                break
            assert (
                hashlib.sha256(source.read_bytes()).hexdigest() == plan["runtime_sha256"]
            ), "Runtime changed after plan freeze"
            assert job["argv"][0] == "timeout" and 0 < int(job["argv"][1]) <= 14400
            holder = subprocess.run(["fuser", "/dev/tenstorrent/0"], capture_output=True, text=True)
            assert (
                holder.returncode == 1 and not holder.stdout.strip()
            ), f"Chip0 is held: {holder.stdout} {holder.stderr}"
            name = job["name"]
            archive = root / "sources" / name
            archive.mkdir(parents=True, exist_ok=True)
            for relative in (
                "tt/fused_decoder.py",
                "tests/run_fused_probe.py",
                "tests/run_fused_attention_probe.py",
                "tests/run_fused_plan.py",
            ):
                shutil.copyfile(model / relative, archive / Path(relative).name)
            row = dict(name=name, status="running")
            progress["rows"].append(row)
            save()
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "models.autoports.qwen_qwen3_8_flash_next.tests.run_recorded",
                    "--record",
                    str(root / f"{name}_provenance.json"),
                    "--log",
                    str(root / "logs" / f"{name}.log"),
                    "--",
                    *job["argv"],
                ],
                env={**environment, **job.get("environment", {})},
            )
            row.update(status="passed" if result.returncode == 0 else "failed", exit_code=result.returncode)
            save()
            if result.returncode:
                progress["status"] = "failed"
                raise SystemExit(result.returncode)
        else:
            progress["status"] = "completed"
    finally:
        save()


if __name__ == "__main__":
    main()
