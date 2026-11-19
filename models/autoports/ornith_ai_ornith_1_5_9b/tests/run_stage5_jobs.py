# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, serialized stage-five experiments with source snapshots and recovery."""

import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    art = root / "doc/optimized_multichip_decoder"
    jobs = json.loads(Path(sys.argv[1]).read_text())
    with open("/home/ttuser/ornith-9b/tt-device-stage5.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for job in jobs:
            if (art / "STOP").exists():
                return 75
            label = job["label"]
            target = art / (label + "_command.json")
            if target.exists():
                raise RuntimeError(f"Refusing to overwrite {target}")
            snapshots = art / "source_snapshots"
            snapshots.mkdir(exist_ok=True)
            hashes = {}
            source_files = list((root / "tt").glob("*.py")) + list((root / "tests").glob("*.py"))
            source_files += [
                p
                for folder in ("tests/kernels", "tt/kernels")
                for p in (root / folder).rglob("*")
                if p.suffix in (".cpp", ".hpp", ".h")
            ]
            for source in sorted(source_files):
                content = source.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                snapshot = snapshots / f"{source.stem}_{digest[:12]}.py.txt"
                if not snapshot.exists():
                    snapshot.write_bytes(content)
                hashes[str(source.relative_to(root))] = digest
            command = ["timeout", str(job.get("timeout", 600)), *job["command"]]
            host_calibration = any("record_precision_inputs" in str(part) for part in command)
            row = dict(
                job,
                command=command,
                source_sha256=hashes,
                regime="serial",
                devices=[] if host_calibration else [0] if "--baseline" in command else [0, 1, 2, 3],
                start=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            target.write_text(json.dumps(row, indent=2) + "\n")
            print("START", label, flush=True)
            begin = time.monotonic()
            with (art / (label + ".log")).open("w") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=job.get("cwd"),
                    env={**os.environ, **job.get("env", {})},
                )
            row.update(
                exit_code=result.returncode,
                seconds=time.monotonic() - begin,
                end=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            with (art / "device_jobs.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
            print("END", label, result.returncode, round(row["seconds"], 2), flush=True)
            if result.returncode:
                recovery = subprocess.run(
                    ["bash", str(root / "tests/recover_stage5.sh"), label + "_recovery"],
                    env={**os.environ, "STAGE5_LOCK_HELD": "1"},
                )
                if recovery.returncode or not job.get("continue_on_failure", False):
                    return recovery.returncode or result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
