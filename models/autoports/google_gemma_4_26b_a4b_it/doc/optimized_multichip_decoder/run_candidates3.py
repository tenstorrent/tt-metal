# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
cases = [
    ("qkv4_persistent_sliding", 0, ["--attention-bfp4", "qkv", "--persistent-ccl"]),
    ("residual_l1_sliding", 0, ["--residual-l1"]),
    ("mmrs_v2_sliding", 0, ["--ring", "--sharded-residual", "--no-grouped-moe-reduce", "--fused-agmm", "--fused-mmrs"]),
    ("qkv4_k44_sliding", 0, ["--attention-bfp4", "qkv", "--projection-k", "44"]),
    ("qkv4_k88_sliding", 0, ["--attention-bfp4", "qkv", "--projection-k", "88"]),
    (
        "qkv4_dram_r1_sliding",
        0,
        ["--attention-bfp4", "qkv", "--attention-dram", "qkv", "--dram-readers", "1", "--dram-storage-cores", "4"],
    ),
    (
        "qkv4_dram_r2_sliding",
        0,
        ["--attention-bfp4", "qkv", "--attention-dram", "qkv", "--dram-readers", "2", "--dram-storage-cores", "4"],
    ),
    (
        "qkv4_dram_r3_sliding",
        0,
        ["--attention-bfp4", "qkv", "--attention-dram", "qkv", "--dram-readers", "3", "--dram-storage-cores", "4"],
    ),
    ("qkv4_persistent_full", 5, ["--attention-bfp4", "qkv", "--persistent-ccl"]),
    ("residual_l1_full", 5, ["--residual-l1"]),
    ("mmrs_v2_full", 5, ["--ring", "--sharded-residual", "--no-grouped-moe-reduce", "--fused-agmm", "--fused-mmrs"]),
    ("qkv4_k44_full", 5, ["--attention-bfp4", "qkv", "--projection-k", "44"]),
    ("qkv4_k88_full", 5, ["--attention-bfp4", "qkv", "--projection-k", "88"]),
    (
        "qkv4_dram_r1_full",
        5,
        ["--attention-bfp4", "qkv", "--attention-dram", "qkv", "--dram-readers", "1", "--dram-storage-cores", "4"],
    ),
    (
        "qkv4_dram_r2_full",
        5,
        ["--attention-bfp4", "qkv", "--attention-dram", "qkv", "--dram-readers", "2", "--dram-storage-cores", "4"],
    ),
    (
        "qkv4_dram_r3_full",
        5,
        ["--attention-bfp4", "qkv", "--attention-dram", "qkv", "--dram-readers", "3", "--dram-storage-cores", "4"],
    ),
]
for name, layer, flags in cases:
    if (root / (name + ".command.json")).exists():
        continue
    if "--dram-readers" in flags and flags[flags.index("--dram-readers") + 1] != "1":
        continue
    cmd = [
        sys.executable,
        "-m",
        "models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder",
        "--no-optimized-decode",
        "--attention-ccl-dtype",
        "bfloat16",
        "--layer",
        str(layer),
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
    with (root / (name + ".log")).open("w") as log:
        result = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "HF_HUB_OFFLINE": "1"}, timeout=600
        )
    (root / (name + ".command.json")).write_text(
        json.dumps({"command": cmd, "exit_code": result.returncode}, indent=2) + "\n"
    )
    print("END", name, result.returncode, flush=True)
    if result.returncode:
        break
