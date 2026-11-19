# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serialize signposted TP4 profiles from a candidate manifest."""

import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("manifest", type=Path)
p.add_argument("--prefix", required=True)
p.add_argument("--default-policy", action="store_true")
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
d = root / "doc/optimized_multichip_decoder"
env = dict(
    os.environ,
    TT_METAL_DEVICE_PROFILER="1",
    TT_METAL_TRACE_ALLOC_TRACKING="0",
    TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
    TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="5000",
)
for key in ("TT_METAL_WATCHER", "TT_METAL_PROFILER_MID_RUN_DUMP"):
    env.pop(key, None)
for job in json.loads(a.manifest.read_text()):
    if (d / "STOP").exists():
        break
    tag = f"{a.prefix}_l{job['layer']}_b{job['batch']}_s{job['length']}"
    output = d / (tag + ".json")
    tracy = d / "tracy" / tag
    perf = d / "perf" / tag
    cmd = [
        "python",
        "-m",
        "tracy",
        "-r",
        "-p",
        "--no-web-server",
        "-o",
        str(tracy),
        "-m",
        "models.autoports.qwen_qwen3_6_27b.tests.run_multichip_decoder",
        "--trace",
        "--profile-drain",
        "--repeats",
        "5",
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
    ]
    if not a.default_policy:
        policy_file = d / (tag + ".policy.json")
        policy_file.write_text(json.dumps(job.get("policy", {}), indent=2) + "\n")
        cmd += ["--policy-file", str(policy_file)]
    if job.get("activation_file"):
        cmd += ["--activation-file", job["activation_file"]]
    subprocess.run(["bash", str(d / "run_job.sh"), tag, *cmd], env=env, check=True)
    subprocess.run(
        [
            "python",
            "-m",
            "models.autoports.qwen_qwen3_6_27b.tests.render_perf",
            "--advice",
            "--plain-artifacts",
            "--tracy-root",
            str(tracy),
            "--result",
            str(output),
            "--output",
            str(perf),
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "models.autoports.qwen_qwen3_6_27b.tests.account_multichip_perf",
            "--result",
            str(output),
            "--perf",
            str(perf),
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "models.autoports.qwen_qwen3_6_27b.tests.account_optimized_multichip_perf",
            "--result",
            str(output),
            "--perf",
            str(perf),
        ],
        env=env,
        check=True,
    )
