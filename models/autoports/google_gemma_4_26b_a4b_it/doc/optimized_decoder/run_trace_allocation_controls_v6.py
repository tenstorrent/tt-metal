# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Classify request-reuse allocation warnings using the runtime tracker."""
import json
import os
import subprocess
from pathlib import Path

p = Path(__file__).resolve().parent
journal = []
for layer in (0, 5):
    stem = p / f"trace_alloc_v6_layer{layer}"
    assert not stem.with_suffix(".log").exists()
    command = [
        "python",
        "-m",
        "models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_contract",
        "--contract",
        "request_reuse",
        "--layer",
        str(layer),
        "--input-fixture",
        str(p / f"actual_text_layer{layer}_4096_128.pt"),
        "--output",
        str(stem.with_suffix(".json")),
    ]
    environment = os.environ | {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "TT_METAL_TRACE_ALLOC_TRACKING": "1",
        "TT_METAL_TRACE_ALLOC_TRACEBACKS": "1",
    }
    assert not environment.get("TT_METAL_WATCHER") and not environment.get("TT_METAL_DEVICE_PROFILER")
    with stem.with_suffix(".log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=600)
    journal.append(
        {
            "command": command,
            "returncode": result.returncode,
            "environment": {k: v for k, v in environment.items() if k.startswith("TT_METAL_TRACE_ALLOC")},
        }
    )
    (p / "trace_alloc_v6_commands.json").write_text(json.dumps(journal, indent=2) + "\n")
    print(layer, result.returncode, flush=True)
