# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serialize bounded stage-5 candidates and retain complete provenance."""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "doc/optimized_multichip_decoder"
p = argparse.ArgumentParser()
p.add_argument("manifest", type=Path)
p.add_argument("--resume", action="store_true")
a = p.parse_args()
env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="0", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
    env.pop(key, None)
for job in json.loads(a.manifest.read_text()):
    if (D / "STOP").exists():
        print("Stopped between jobs", flush=True)
        break
    output = D / (job["name"] + ".json")
    if a.resume and output.exists() and json.loads(output.read_text()).get("pass"):
        continue
    cmd = [
        "python",
        "-m",
        job.get("module", "models.autoports.qwen_qwen3_6_27b.tests.run_multichip_decoder"),
        *job.get("module_args", []),
        "--trace",
        "--repeats",
        "100",
        "--layer",
        str(job["layer"]),
        "--batch",
        str(job["batch"]),
        "--length",
        str(job["length"]),
        "--reference",
        job["reference"],
        "--output",
        str(output),
        "--policy-json",
        json.dumps(job.get("policy", {})),
    ]
    if job.get("activation_file"):
        cmd += ["--activation-file", job["activation_file"]]
    if job.get("baseline"):
        cmd += ["--baseline"]
    if job.get("legacy_decode_layout"):
        cmd += ["--legacy-decode-layout"]
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start = time.monotonic()
    sha = hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest()
    code = subprocess.run(["bash", str(D / "run_job.sh"), job["name"], *cmd], env=env).returncode
    record = dict(
        job=job,
        command=cmd,
        utc=utc,
        elapsed_seconds=time.monotonic() - start,
        exit_code=code,
        source_sha256=sha,
        regime="serial",
        chips=[0, 1, 2, 3],
    )
    with (D / "candidate_commands.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps({k: record[k] for k in ("utc", "elapsed_seconds", "exit_code", "source_sha256")}), flush=True)
    if code:
        raise SystemExit(code)
