# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recheck affected full-attention tails and the tight K128 branch on v7."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent
RUNTIME = DOC.parent.parent / "tt/optimized_decoder.py"
EXPECTED = "daa82a4a5197a007ccd5d29d912f082e695fd37a596578f25fba96a6694e625b"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    journal = DOC / "v7_boundary_commands.json"
    assert not journal.exists()
    jobs = []
    tight = json.loads((DOC / "tight_cache_manual_commands.json").read_text())[1]
    command = [part.replace("tight_cache_v6", "tight_cache_v7") for part in tight["command"]]
    jobs.append(dict(command=command, watcher=True, kind="tight_cache"))
    for entry in json.loads((DOC / "prefill_boundary_commands.json").read_text()):
        command = entry["command"]
        if command[command.index("--layer") + 1] != "5" or not any("run_optimized_contract" in v for v in command):
            continue
        command = [part.replace("prefill_boundary_optimized_", "prefill_boundary_v7_optimized_") for part in command]
        jobs.append(
            dict(
                command=command,
                watcher=False,
                kind="warmed_prefill_boundary",
                logical_length=entry["logical_length"],
                physical_rows=entry["physical_rows"],
            )
        )
    assert len(jobs) == 5
    records = []
    for job in jobs:
        assert digest(RUNTIME) == EXPECTED
        command = job["command"]
        command[0] = sys.executable
        output = Path(command[command.index("--output") + 1])
        assert not output.exists()
        environment = os.environ | {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        if job["watcher"]:
            environment["TT_METAL_WATCHER"] = "10"
        print("Starting", output.stem, flush=True)
        with output.with_suffix(".log").open("w") as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=1200)
        if job["watcher"]:
            shutil.copyfile("generated/watcher/watcher.log", output.with_suffix(".device.log"))
        job.update(returncode=run.returncode, runtime_sha256=EXPECTED, driver_sha256=digest(Path(__file__)))
        if output.exists():
            job["output_sha256"] = digest(output)
        records.append(job)
        journal.write_text(json.dumps(records, indent=2) + "\n")
        assert digest(RUNTIME) == EXPECTED
        print("Completed", output.stem, run.returncode, flush=True)
        if run.returncode:
            raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
