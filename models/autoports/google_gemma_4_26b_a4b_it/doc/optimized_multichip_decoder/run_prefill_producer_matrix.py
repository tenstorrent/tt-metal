# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Close final profiler advice with serialized, real-shape producer-L1 controls."""

import json
import statistics
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for layer, kind in ((0, "sliding"), (5, "full")):
    for role in ("control", "qkv", "output", "router", "shared"):
        name = f"prefill_producer_{role}_{kind}"
        command = [
            sys.executable,
            "-m",
            "models.autoports.google_gemma_4_26b_a4b_it.doc.optimized_multichip_decoder.probe_prefill_producer_l1",
            "--prefill-l1-role",
            role,
            "--layer",
            str(layer),
            "--length",
            "4096",
            "--steps",
            "128",
            "--trace",
            "--check-cache",
            "--prefill-timing-samples",
            "5",
            "--output",
            str(root / (name + ".json")),
        ]
        print("START", name, flush=True)
        with (root / (name + ".log")).open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        (root / (name + ".command.json")).write_text(
            json.dumps(dict(command=command, exit_code=result.returncode), indent=2) + "\n"
        )
        print("END", name, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
        data = json.loads((root / (name + ".json")).read_text())
        print(
            "RESULT",
            {phase: statistics.median(values) for phase, values in data["timings"]["4"].items()},
            min(data["pcc"]),
            flush=True,
        )
