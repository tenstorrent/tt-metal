# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recheck material geometry/topology after adjacent-stack precision selection."""

import json
import statistics
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sharded = ["--ring", "--sharded-residual", "--no-grouped-moe-reduce", "--fused-agmm", "--sharded-moe-bfp8"]
cases = [
    ("selected_control", []),
    ("precision_matched_baseline", ["--no-optimized-decode", "--expert-gate-dtype", "bfloat8_b"]),
    ("qkv_k22", ["--projection-k", "22"]),
    ("qkv_k88", ["--projection-k", "88"]),
    ("qkv_n1", ["--dense-geometry", "qkv-n1"]),
    ("qkv_n4", ["--dense-geometry", "qkv-n4"]),
    ("expert_n1_k88", ["--sparse-gate-geometry", "n1-k88"]),
    ("expert_n2_k44", ["--sparse-gate-geometry", "n2-k44"]),
    ("expert_n2_k88", ["--sparse-gate-geometry", "n2-k88"]),
    ("expert_hifi2", ["--expert-fidelity", "HiFi2"]),
    ("expert_separate", ["--split-expert-gate"]),
    ("qkv_dram_r2", ["--attention-dram", "qkv", "--dram-readers", "2"]),
    ("residual_l1", ["--residual-l1"]),
    ("sharded_qkv_agmm", sharded),
    ("sharded_wo_agmm", sharded + ["--output-agmm"]),
    ("sharded_mmrs", sharded + ["--fused-mmrs"]),
]
for suffix, flags in cases:
    name = "final_policy_" + suffix
    record = root / (name + ".command.json")
    if record.exists() and json.loads(record.read_text())["exit_code"] == 0:
        continue
    command = [
        sys.executable,
        "-m",
        "models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder",
        "--layer",
        "0",
        "--length",
        "4096",
        "--steps",
        "128",
        "--trace",
        "--check-cache",
        "--output",
        str(root / (name + ".json")),
    ] + flags
    print("START", name, flush=True)
    with (root / (name + ".log")).open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
    record.write_text(json.dumps(dict(command=command, exit_code=result.returncode), indent=2) + "\n")
    print("END", name, result.returncode, flush=True)
    if result.returncode:
        break
    data = json.loads((root / (name + ".json")).read_text())
    print("RESULT", statistics.median(data["timings"]["4"]["decode_host_us"]), min(data["pcc"]), flush=True)
