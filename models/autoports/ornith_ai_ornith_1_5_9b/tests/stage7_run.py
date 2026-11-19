# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial stage evidence command wrapper; callers supply timeout explicitly."""

import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    name, *command = sys.argv[1:]
    if command[0] != "timeout":
        raise ValueError("Every device command requires an explicit timeout")
    root = Path(__file__).resolve().parents[1]
    evidence = root / "doc/optimized_full_model"
    log = evidence / "logs" / (name + ".log")
    record = {
        "name": name,
        "cwd": os.getcwd(),
        "command": command,
        "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "regime": "serial devices 0-3",
        "environment": {k: v for k, v in os.environ.items() if k.startswith(("TT_", "TRACY_", "OMP_", "READINESS_"))},
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("tt", "tests")
            for p in sorted((root / folder).rglob("*"))
            if p.is_file() and p.suffix in (".py", ".cpp", ".hpp")
        },
        "log": str(log.relative_to(root)),
    }
    with log.open("w") as stream:
        status = subprocess.run(command, check=False, stdout=stream, stderr=subprocess.STDOUT).returncode
    record.update(exit_code=status, finish_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (evidence / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(name, "exit", status, "log", log, flush=True)
    sys.exit(status)


if __name__ == "__main__":
    main()
