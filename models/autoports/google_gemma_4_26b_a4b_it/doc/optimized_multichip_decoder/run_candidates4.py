# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
cases = [
    (
        "fixed_dram_qkv_bfp8_r2_sliding",
        0,
        ["--attention-dram", "qkv", "--dram-readers", "2", "--dram-storage-cores", "4"],
    ),
    (
        "fixed_dram_qkv_bfp8_r3_sliding",
        0,
        ["--attention-dram", "qkv", "--dram-readers", "3", "--dram-storage-cores", "4"],
    ),
    (
        "fixed_dram_qkv_bfp4_r2_sliding",
        0,
        ["--attention-dram", "qkv", "--dram-readers", "2", "--dram-storage-cores", "4", "--attention-bfp4", "qkv"],
    ),
    (
        "fixed_dram_qkv_bfp4_r3_v2_sliding",
        0,
        ["--attention-dram", "qkv", "--dram-readers", "3", "--dram-storage-cores", "4", "--attention-bfp4", "qkv"],
    ),
    (
        "fixed_dram_output_bfp8_r2_sliding",
        0,
        ["--attention-dram", "output", "--dram-readers", "2", "--dram-storage-cores", "4"],
    ),
    (
        "fixed_dram_output_bfp8_r3_sliding",
        0,
        ["--attention-dram", "output", "--dram-readers", "3", "--dram-storage-cores", "4"],
    ),
    (
        "fixed_dram_output_bfp4_r2_sliding",
        0,
        [
            "--attention-dram",
            "output",
            "--dram-readers",
            "2",
            "--dram-storage-cores",
            "4",
            "--attention-bfp4",
            "output",
        ],
    ),
    (
        "fixed_dram_output_bfp4_r3_sliding",
        0,
        [
            "--attention-dram",
            "output",
            "--dram-readers",
            "3",
            "--dram-storage-cores",
            "4",
            "--attention-bfp4",
            "output",
        ],
    ),
    ("fixed_shared_dram_r1_sliding", 0, ["--shared-dram", "--shared-geometry", "0", "--dram-readers", "1"]),
    ("fixed_shared_dram_r2_sliding", 0, ["--shared-dram", "--shared-geometry", "0", "--dram-readers", "2"]),
    ("fixed_shared_dram_r3_sliding", 0, ["--shared-dram", "--shared-geometry", "0", "--dram-readers", "3"]),
    (
        "residual_cumulative_sliding",
        0,
        ["--residual-l1", "--attention-bfp4", "qkv", "--persistent-ccl", "--projection-k", "44"],
    ),
    ("fixed_dram_qkv_bfp8_r2_full", 5, ["--attention-dram", "qkv", "--dram-readers", "2", "--dram-storage-cores", "4"]),
    ("fixed_dram_qkv_bfp8_r3_full", 5, ["--attention-dram", "qkv", "--dram-readers", "3", "--dram-storage-cores", "4"]),
    (
        "fixed_dram_qkv_bfp4_r2_full",
        5,
        ["--attention-dram", "qkv", "--dram-readers", "2", "--dram-storage-cores", "4", "--attention-bfp4", "qkv"],
    ),
    (
        "fixed_dram_qkv_bfp4_r3_full",
        5,
        ["--attention-dram", "qkv", "--dram-readers", "3", "--dram-storage-cores", "4", "--attention-bfp4", "qkv"],
    ),
    (
        "fixed_dram_output_bfp8_r2_full",
        5,
        ["--attention-dram", "output", "--dram-readers", "2", "--dram-storage-cores", "4"],
    ),
    (
        "fixed_dram_output_bfp8_r3_full",
        5,
        ["--attention-dram", "output", "--dram-readers", "3", "--dram-storage-cores", "4"],
    ),
    (
        "fixed_dram_output_bfp4_r2_full",
        5,
        [
            "--attention-dram",
            "output",
            "--dram-readers",
            "2",
            "--dram-storage-cores",
            "4",
            "--attention-bfp4",
            "output",
        ],
    ),
    (
        "fixed_dram_output_bfp4_r3_full",
        5,
        [
            "--attention-dram",
            "output",
            "--dram-readers",
            "3",
            "--dram-storage-cores",
            "4",
            "--attention-bfp4",
            "output",
        ],
    ),
    ("fixed_shared_dram_r1_full", 5, ["--shared-dram", "--shared-geometry", "0", "--dram-readers", "1"]),
    ("fixed_shared_dram_r2_full", 5, ["--shared-dram", "--shared-geometry", "0", "--dram-readers", "2"]),
    ("fixed_shared_dram_r3_full", 5, ["--shared-dram", "--shared-geometry", "0", "--dram-readers", "3"]),
    (
        "residual_cumulative_full",
        5,
        ["--residual-l1", "--attention-bfp4", "both", "--persistent-ccl", "--projection-k", "44"],
    ),
]
for name, layer, flags in cases:
    if (root / (name + ".command.json")).exists() and json.loads((root / (name + ".command.json")).read_text())[
        "exit_code"
    ] == 0:
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
