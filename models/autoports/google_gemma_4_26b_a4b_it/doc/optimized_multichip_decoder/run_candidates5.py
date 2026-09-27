# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
cases = [
    ("cumulative_sliding", 0, ["--attention-bfp4", "qkv", "--projection-k", "44", "--persistent-ccl"]),
    ("ccl_l1_sliding", 0, ["--attention-bfp4", "qkv", "--projection-k", "44", "--ccl-l1"]),
    (
        "ccl_l1_persistent_sliding",
        0,
        ["--attention-bfp4", "qkv", "--projection-k", "44", "--ccl-l1", "--persistent-ccl"],
    ),
    ("qkv_hifi2_sliding", 0, ["--attention-bfp4", "qkv", "--projection-k", "44", "--qkv-fidelity", "HiFi2"]),
    ("qkv_n4_sliding", 0, ["--attention-bfp4", "qkv", "--projection-k", "44", "--dense-geometry", "qkv-n4"]),
    (
        "sharded_precision_sliding",
        0,
        [
            "--attention-bfp4",
            "qkv",
            "--ring",
            "--sharded-residual",
            "--no-grouped-moe-reduce",
            "--fused-agmm",
            "--sharded-moe-bfp8",
        ],
    ),
    (
        "mmrs_precision_sliding",
        0,
        [
            "--attention-bfp4",
            "qkv",
            "--ring",
            "--sharded-residual",
            "--no-grouped-moe-reduce",
            "--fused-agmm",
            "--sharded-moe-bfp8",
            "--fused-mmrs",
        ],
    ),
    ("cumulative_full", 5, ["--attention-bfp4", "both", "--projection-k", "44", "--persistent-ccl"]),
    ("ccl_l1_full", 5, ["--attention-bfp4", "both", "--projection-k", "44", "--ccl-l1"]),
    ("ccl_l1_persistent_full", 5, ["--attention-bfp4", "both", "--projection-k", "44", "--ccl-l1", "--persistent-ccl"]),
    ("qkv_hifi2_full", 5, ["--attention-bfp4", "both", "--projection-k", "44", "--qkv-fidelity", "HiFi2"]),
    ("qkv_n4_full", 5, ["--attention-bfp4", "both", "--projection-k", "44", "--dense-geometry", "qkv-n4"]),
    (
        "sharded_precision_full",
        5,
        [
            "--attention-bfp4",
            "both",
            "--ring",
            "--sharded-residual",
            "--no-grouped-moe-reduce",
            "--fused-agmm",
            "--sharded-moe-bfp8",
        ],
    ),
    (
        "mmrs_precision_full",
        5,
        [
            "--attention-bfp4",
            "both",
            "--ring",
            "--sharded-residual",
            "--no-grouped-moe-reduce",
            "--fused-agmm",
            "--sharded-moe-bfp8",
            "--fused-mmrs",
        ],
    ),
    ("output_hifi2_bfp4_full", 5, ["--attention-bfp4", "both", "--persistent-ccl", "--output-fidelity", "HiFi2"]),
    ("output_n4_bfp4_full", 5, ["--attention-bfp4", "both", "--persistent-ccl", "--dense-geometry", "output-n4"]),
    (
        "output_k32_bfp4_full",
        5,
        ["--attention-bfp4", "both", "--persistent-ccl", "--projection-role", "output", "--projection-k", "32"],
    ),
    (
        "output_k64_bfp4_full",
        5,
        ["--attention-bfp4", "both", "--persistent-ccl", "--projection-role", "output", "--projection-k", "64"],
    ),
    ("qkv_n1_bfp4_sliding", 0, ["--attention-bfp4", "qkv", "--dense-geometry", "qkv-n1", "--projection-k", "44"]),
    ("qkv_n1_bfp4_v2_full", 5, ["--attention-bfp4", "both", "--dense-geometry", "qkv-n1", "--projection-k", "44"]),
    ("output_n2_bfp4_full", 5, ["--attention-bfp4", "both", "--persistent-ccl", "--dense-geometry", "output-n2"]),
    (
        "output_k16_bfp4_full",
        5,
        ["--attention-bfp4", "both", "--persistent-ccl", "--projection-role", "output", "--projection-k", "16"],
    ),
    (
        "fixed_dram_output_bfp4_r1_full",
        5,
        ["--attention-bfp4", "output", "--attention-dram", "output", "--dram-storage-cores", "4"],
    ),
    (
        "output_agmm_sliding",
        0,
        [
            "--attention-bfp4",
            "qkv",
            "--ring",
            "--sharded-residual",
            "--no-grouped-moe-reduce",
            "--fused-agmm",
            "--sharded-moe-bfp8",
            "--output-agmm",
        ],
    ),
    (
        "output_agmm_full",
        5,
        [
            "--attention-bfp4",
            "both",
            "--ring",
            "--sharded-residual",
            "--no-grouped-moe-reduce",
            "--fused-agmm",
            "--sharded-moe-bfp8",
            "--output-agmm",
        ],
    ),
    (
        "cumulative_activation_attention_sliding",
        0,
        ["--attention-bfp4", "qkv", "--persistent-ccl", "--projection-k", "44", "--activation-bfp8", "attention"],
    ),
    (
        "cumulative_activation_shared_sliding",
        0,
        ["--attention-bfp4", "qkv", "--persistent-ccl", "--projection-k", "44", "--activation-bfp8", "shared"],
    ),
    (
        "cumulative_activation_attention_full",
        5,
        ["--attention-bfp4", "both", "--persistent-ccl", "--projection-k", "44", "--activation-bfp8", "attention"],
    ),
    (
        "cumulative_activation_shared_full",
        5,
        ["--attention-bfp4", "both", "--persistent-ccl", "--projection-k", "44", "--activation-bfp8", "shared"],
    ),
    (
        "sharded_precision_sliding_persistent",
        0,
        [
            "--attention-bfp4",
            "qkv",
            "--ring",
            "--sharded-residual",
            "--no-grouped-moe-reduce",
            "--fused-agmm",
            "--sharded-moe-bfp8",
            "--persistent-ccl",
        ],
    ),
    (
        "sharded_precision_full_persistent",
        5,
        [
            "--attention-bfp4",
            "both",
            "--ring",
            "--sharded-residual",
            "--no-grouped-moe-reduce",
            "--fused-agmm",
            "--sharded-moe-bfp8",
            "--persistent-ccl",
        ],
    ),
    (
        "output_agmm_sliding_persistent",
        0,
        [
            "--attention-bfp4",
            "qkv",
            "--ring",
            "--sharded-residual",
            "--no-grouped-moe-reduce",
            "--fused-agmm",
            "--sharded-moe-bfp8",
            "--output-agmm",
            "--persistent-ccl",
        ],
    ),
    (
        "output_agmm_full_persistent",
        5,
        [
            "--attention-bfp4",
            "both",
            "--ring",
            "--sharded-residual",
            "--no-grouped-moe-reduce",
            "--fused-agmm",
            "--sharded-moe-bfp8",
            "--output-agmm",
            "--persistent-ccl",
        ],
    ),
    (
        "sharded_qkv4_control_full",
        5,
        [
            "--attention-bfp4",
            "qkv",
            "--ring",
            "--sharded-residual",
            "--no-grouped-moe-reduce",
            "--fused-agmm",
            "--sharded-moe-bfp8",
        ],
    ),
    (
        "sharded_qkv4_control_mmrs_full",
        5,
        [
            "--attention-bfp4",
            "qkv",
            "--ring",
            "--sharded-residual",
            "--no-grouped-moe-reduce",
            "--fused-agmm",
            "--sharded-moe-bfp8",
            "--fused-mmrs",
        ],
    ),
]
for name, layer, flags in cases:
    if (root / (name + ".command.json")).exists() and (
        (root / (name + ".json")).exists()
        or json.loads((root / (name + ".command.json")).read_text())["exit_code"] == 0
    ):
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
    if result.returncode and not (root / (name + ".json")).exists():
        break
