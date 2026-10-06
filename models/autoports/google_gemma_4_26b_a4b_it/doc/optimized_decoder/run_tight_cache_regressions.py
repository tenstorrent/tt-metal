# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise both native K-block branches at the public 128-token cache boundary."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch

p = Path(__file__).resolve().parent
source_path = p / "actual_text_layer5_4096_128.pt"
source = torch.load(source_path, map_location="cpu", weights_only=True)
journal = []
for length, steps in [(1088, 1), (1089, 1), (1152, 0), (1152, 1), (1153, 1), (1281, 1), (2049, 1)]:
    name = f"tight_cache_v6_layer5_{length}_steps{steps}"
    fixture = p / (name + ".pt")
    output = p / (name + ".json")
    assert not output.exists()
    metadata = dict(
        source["metadata"],
        length=length,
        steps=steps,
        slice_source=str(source_path),
        slice_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
    torch.save(
        dict(
            metadata=metadata,
            prefill=source["prefill"][:, :length].clone(),
            decode=source["prefill"][:, length : length + steps].clone(),
        ),
        fixture,
    )
    capacity = ((length + steps + 127) // 128) * 128
    command = [
        "python",
        "-m",
        "models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_cache_capacity",
        "--contract",
        "run_decoder",
        "--layer",
        "5",
        "--length",
        str(length),
        "--real",
        "--input-fixture",
        str(fixture),
        "--steps",
        str(steps),
        "--cache-extent",
        str(capacity),
        "--verify-program-cache",
        "--output",
        str(output),
    ]
    if steps:
        command.append("--decode")
    watcher = length == 1281
    env = os.environ | {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    if watcher:
        env["TT_METAL_WATCHER"] = "10"
    with (p / (name + ".log")).open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=600)
    if watcher:
        shutil.copyfile("generated/watcher/watcher.log", p / (name + ".device.log"))
    journal.append(
        dict(
            command=command,
            returncode=result.returncode,
            watcher=watcher,
            length=length,
            steps=steps,
            capacity=capacity,
        )
    )
    (p / "tight_cache_v6_commands.json").write_text(json.dumps(journal, indent=2) + "\n")
    print(name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
