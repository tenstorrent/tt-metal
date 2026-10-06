# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for name, layer, flags in [
    ("final_payload_attention8_sliding", 0, ["--attention-ccl-dtype", "bfloat8_b"]),
    ("final_payload_moe8_full", 5, ["--moe-ccl-bfp8"]),
]:
    command = [
        sys.executable,
        "-m",
        "models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder",
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
            command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "HF_HUB_OFFLINE": "1"}, timeout=600
        )
    (root / (name + ".command.json")).write_text(
        json.dumps(dict(command=command, exit_code=result.returncode), indent=2) + "\n"
    )
    print("END", name, result.returncode, flush=True)
    if result.returncode and not (root / (name + ".json")).exists():
        break
