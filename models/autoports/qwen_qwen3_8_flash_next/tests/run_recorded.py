# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run one authorized stage command and retain exact source/command provenance.

This host-only wrapper never opens hardware or changes the supplied environment.
Use an explicit timeout in the child command. Exit status is propagated.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A child command is required")
    root = Path(__file__).resolve().parents[1]
    sources = sorted(
        path
        for directory in (root / "tt", root / "tests")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in (".py", ".cpp", ".hpp", ".h")
    )
    record = {
        "argv": command,
        "cwd": os.getcwd(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TT_VISIBLE_DEVICES",
                "TT_MESH_GRAPH_DESC_PATH",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_WATCHER",
                "TT_METAL_LOGS_PATH",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY",
                "TT_METAL_STREAMING_PROFILER",
                "TT_METAL_STREAMING_PROFILER_DRAM_MB",
                "TT_METAL_STREAMING_PROFILER_FIFO_MB",
                "TT_METAL_STREAMING_PROFILER_OPS_CSV",
                "TT_METAL_STREAMING_PROFILER_ZONE_CSV",
                "TT_METAL_STREAMING_PROFILER_TRACY",
                "PYTHONPATH",
            )
        },
        "source_sha256": {str(path.relative_to(root)): digest(path) for path in sources},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "log": str(args.log),
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "running",
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()
    with args.log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        # Append a stable completion marker for wait_for; the provenance JSON
        # is rewritten, so an incremental log follower cannot watch its status.
        stream.write(f"\nRUN_RECORDED_FINISHED exit_code={result.returncode}\n")
    record.update(
        status="finished",
        exit_code=result.returncode,
        elapsed_seconds=time.monotonic() - started,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        log_sha256=digest(args.log),
    )
    args.record.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ("status", "exit_code", "elapsed_seconds", "log")}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
