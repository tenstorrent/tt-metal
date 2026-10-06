# SPDX-License-Identifier: Apache-2.0
"""Execute an explicit JSON job list serially, preserving commands and stop-on-error."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .multichip_sweep import OUT

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    a = p.parse_args()
    for job in json.load(open(a.jobs)):
        command = [
            sys.executable,
            "-m",
            "models.autoports.aleph_alpha_kolibri_1_bf16.tests." + job["module"],
            *job["args"],
        ]
        env = os.environ | job.get("env", {})
        log = OUT / job["log"]
        sources = {}
        snapshot = OUT / "source_snapshots"
        snapshot.mkdir(exist_ok=True)
        for source in sorted(Path(__file__).resolve().parents[1].glob("tt/*.py")) + sorted(
            Path(__file__).resolve().parent.glob("*multichip*.py")
        ):
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            target = snapshot / (digest + ".py.txt")
            if not target.exists():
                target.write_bytes(source.read_bytes())
            sources[str(source)] = digest
        with log.open("w") as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env)
        row = dict(
            command=command,
            environment=job.get("env", {}),
            log=str(log),
            exit_code=result.returncode,
            time=time.time(),
            sources=sources,
        )
        with (OUT / "validation_commands.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(job["log"], result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
