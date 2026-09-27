# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
cases = [
    ("prefill_k17_h4_sliding", 0, ["--shared-prefill-k", "17", "--shared-prefill-subblock", "4"]),
    (
        "prefill_k17_h4_l1_sliding",
        0,
        ["--shared-prefill-k", "17", "--shared-prefill-subblock", "4", "--shared-prefill-l1"],
    ),
    ("prefill_k17_h4_full", 5, ["--shared-prefill-k", "17", "--shared-prefill-subblock", "4"]),
    (
        "prefill_k17_h4_l1_full",
        5,
        ["--shared-prefill-k", "17", "--shared-prefill-subblock", "4", "--shared-prefill-l1"],
    ),
    (
        "split_qkv4_sliding",
        0,
        ["--attention-bfp4", "qkv", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--split-qkv"],
    ),
    (
        "split_qkv4_full",
        5,
        ["--attention-bfp4", "both", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--split-qkv"],
    ),
    (
        "residual_ccl_l1_sliding",
        0,
        ["--attention-bfp4", "qkv", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--residual-l1"],
    ),
    (
        "residual_ccl_l1_full",
        5,
        ["--attention-bfp4", "both", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--residual-l1"],
    ),
    (
        "ccl_worker1_sliding",
        0,
        ["--attention-bfp4", "qkv", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--ccl-workers", "1"],
    ),
    (
        "ccl_worker2_sliding",
        0,
        ["--attention-bfp4", "qkv", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--ccl-workers", "2"],
    ),
    (
        "ccl_worker4_sliding",
        0,
        ["--attention-bfp4", "qkv", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--ccl-workers", "4"],
    ),
    (
        "ccl_buffer2_sliding",
        0,
        ["--attention-bfp4", "qkv", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--ccl-buffers", "2"],
    ),
    (
        "ccl_chunk1_v2_sliding",
        0,
        ["--attention-bfp4", "qkv", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--ccl-chunks", "1"],
    ),
    (
        "ccl_chunk4_sliding",
        0,
        ["--attention-bfp4", "qkv", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--ccl-chunks", "4"],
    ),
    (
        "ccl_worker1_full",
        5,
        ["--attention-bfp4", "both", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--ccl-workers", "1"],
    ),
    (
        "ccl_worker2_full",
        5,
        ["--attention-bfp4", "both", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--ccl-workers", "2"],
    ),
    (
        "ccl_worker4_full",
        5,
        ["--attention-bfp4", "both", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--ccl-workers", "4"],
    ),
    (
        "ccl_buffer2_full",
        5,
        ["--attention-bfp4", "both", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--ccl-buffers", "2"],
    ),
    (
        "ccl_chunk1_full",
        5,
        ["--attention-bfp4", "both", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--ccl-chunks", "1"],
    ),
    (
        "ccl_chunk4_full",
        5,
        ["--attention-bfp4", "both", "--persistent-ccl", "--ccl-l1", "--projection-k", "44", "--ccl-chunks", "4"],
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
