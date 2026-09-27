# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize current-policy, carried-residual fused families after the K-block fix."""

import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for layer, kind in ((0, "sliding"), (5, "full")):
    for family, flags in (("qkv_agmm", []), ("wo_agmm", ["--output-agmm"]), ("mmrs", ["--fused-mmrs"])):
        name = f"repaired_sharded_{family}_{kind}"
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
            "--ring",
            "--sharded-residual",
            "--no-grouped-moe-reduce",
            "--fused-agmm",
            "--sharded-moe-bfp8",
            "--output",
            str(root / (name + ".json")),
            *flags,
        ]
        print("START", name, flush=True)
        with (root / (name + ".log")).open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        (root / (name + ".command.json")).write_text(
            json.dumps(
                dict(
                    command=command,
                    exit_code=result.returncode,
                    runtime_sha256=hashlib.sha256(
                        (root.parents[1] / "tt/multichip_decoder.py").read_bytes()
                    ).hexdigest(),
                ),
                indent=2,
            )
            + "\n"
        )
        print("END", name, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
        data = json.loads((root / (name + ".json")).read_text())
        print("RESULT", statistics.median(data["timings"]["4"]["decode_host_us"]), min(data["pcc"]), flush=True)
