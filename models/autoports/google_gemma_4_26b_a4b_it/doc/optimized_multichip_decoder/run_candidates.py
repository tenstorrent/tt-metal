# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
cases = [
    ("output_bfp4_sliding", 0, ["--attention-bfp4", "output"]),
    ("output_bfp4_full", 5, ["--attention-bfp4", "output"]),
    ("split_expert_final_sliding", 0, ["--split-expert-gate"]),
    ("split_shared_final_sliding", 0, ["--split-shared-gate"]),
    ("shared_down_bfp4_sliding", 0, ["--shared-down-bfp4"]),
    ("prefill_k17_sliding", 0, ["--shared-prefill-k", "17"]),
    ("prefill_k17_full", 5, ["--shared-prefill-k", "17"]),
    ("prefill_l1_sliding", 0, ["--shared-prefill-l1"]),
    ("prefill_l1_full", 5, ["--shared-prefill-l1"]),
]
for name, layer, flags in cases:
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
