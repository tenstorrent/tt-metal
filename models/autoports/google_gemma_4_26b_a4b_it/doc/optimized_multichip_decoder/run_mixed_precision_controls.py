# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize independent precision controls on the failing actual-context stack."""

import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for name, flag in [
    ("expert_activation16", "--probe-expert-activation-bf16"),
    ("qkv_hifi2", "--probe-qkv-hifi2"),
    ("output_hifi4", "--probe-output-hifi4"),
]:
    name = "stack_mixed_sliding_" + name
    output = root / (name + ".json")
    command = [
        sys.executable,
        "-m",
        "models.autoports.google_gemma_4_26b_a4b_it.doc.optimized_multichip_decoder.probe_stack_precision",
        "--layers",
        "0",
        "5",
        "--attention-precision",
        "baseline",
        "--probe-layers",
        "0",
        "--length",
        "4096",
        "--steps",
        "128",
        flag,
        "--output",
        str(output),
    ]
    print("START", name, flush=True)
    with (root / (name + ".log")).open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    (root / (name + ".command.json")).write_text(
        json.dumps(dict(command=command, exit_code=result.returncode), indent=2) + "\n"
    )
    if not output.exists():
        raise RuntimeError(f"{name}: no numerical result, inspect and adapt the failed control")
    data = json.loads(output.read_text())
    print("END", name, result.returncode, "minimum_pcc", min(row["pcc"] for row in data["comparisons"]), flush=True)
