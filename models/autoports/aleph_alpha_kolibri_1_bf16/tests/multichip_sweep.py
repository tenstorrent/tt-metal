# SPDX-License-Identifier: Apache-2.0
"""Serial fresh-process policy measurements; stop on failure for diagnosis."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "doc/multichip_decoder"
OUT = Path(os.environ.get("MC_ARTIFACT_DIR", str(REFERENCE)))


def run(a):
    rows = json.loads((OUT / "geometry_candidates.json").read_text())
    for row in rows:
        if a.names and row["name"] not in a.names.split(","):
            continue
        name = row["name"]
        command = [
            sys.executable,
            "-m",
            "models.autoports.aleph_alpha_kolibri_1_bf16.tests.multichip_checks",
            "--layer",
            str(a.layer),
            "--tag",
            name,
            "--repetitions",
            str(a.repetitions),
        ]
        env = os.environ | {"MC_POLICY": json.dumps(row["policy"]), "TT_METAL_TRACE_ALLOC_TRACKING": "0"}
        with (OUT / f"{name}_{a.layer}_128.log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        entry = dict(command=command, policy=row["policy"], exit_code=result.returncode, time=time.time())
        with (OUT / "sweep_commands.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(name, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--names", default="")
    p.add_argument("--repetitions", type=int, default=40)
    run(p.parse_args())
