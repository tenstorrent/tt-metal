# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize bounded stage jobs; STOP prevents the next launch, never kills one."""

import datetime
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"
    jobs = json.loads(Path(sys.argv[1]).read_text())
    with open("/home/ttuser/ornith-9b/tt-device-stage5.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for job in jobs:
            if (root / "STOP").exists():
                print("STOP: no further jobs launched", flush=True)
                return 75
            log = root / (job["name"] + ".log")
            if log.exists():
                raise RuntimeError(f"Refusing to overwrite {log}")
            env = dict(os.environ)
            env.update(
                TT_METAL_TRACE_ALLOC_TRACKING="0",
                TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
                OMP_NUM_THREADS="8",
                READINESS_TRACE_REGION_SIZE=str(256 * 1024 * 1024),
            )
            env.update(job.get("env", {}))
            command = ["timeout", str(job.get("timeout", 1800)), *job["command"]]
            row = dict(
                job,
                argv=command,
                started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                regime="serial",
                chips=[0, 1, 2, 3],
            )
            print("START", job["name"], flush=True)
            with log.open("w") as stream:
                row["exit_code"] = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
            row["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with (root / "commands.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print("END", job["name"], row["exit_code"], flush=True)
            if row["exit_code"]:
                return row["exit_code"]
    return 0


if __name__ == "__main__":
    sys.exit(main())
