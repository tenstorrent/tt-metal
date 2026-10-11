# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized terminal-only candidates; decoder policy is invariant."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
DOC = ROOT / "doc/optimized_full_model"
candidates = [
    ("bf16_control", dict(weight_dtype="bfloat16", fidelity="HiFi4")),
    ("bf16_split4_r1_k2", dict(weight_dtype="bfloat16", fidelity="HiFi4", splits=4, readers=1, block_w=2)),
    ("bf16_split4_r2_k4", dict(weight_dtype="bfloat16", fidelity="HiFi4", splits=4, readers=2, block_w=4)),
    ("bf16_split3_r3_k4", dict(weight_dtype="bfloat16", fidelity="HiFi4", splits=3, readers=3, block_w=4)),
    ("bfp8_hifi2_r3_k1", dict(weight_dtype="bfloat8_b", fidelity="HiFi2")),
    ("bfp8_hifi2_r2_k1", dict(weight_dtype="bfloat8_b", fidelity="HiFi2", readers=2)),
    ("bfp8_lofi_r3_k2", dict(weight_dtype="bfloat8_b", fidelity="LoFi", block_w=2)),
    ("bfp8_hifi2_split4_r2_k8", dict(weight_dtype="bfloat8_b", fidelity="HiFi2", splits=4, readers=2, block_w=8)),
    ("bfp8_hifi2_split4_r2_k16", dict(weight_dtype="bfloat8_b", fidelity="HiFi2", splits=4, readers=2, block_w=16)),
]
parser = argparse.ArgumentParser()
parser.add_argument("--start-at", type=int, default=0)
args = parser.parse_args()
rows = json.loads((DOC / "terminal_sweep.json").read_text()) if args.start_at else []
for name, policy in candidates[args.start_at :]:
    destination = DOC / ("terminal_" + name + ".json")
    command = [
        sys.executable,
        str(ROOT / "tests/probe_full_terminal.py"),
        "--policy",
        json.dumps(policy),
        "--output",
        str(destination),
    ]
    print("CANDIDATE_START", name, flush=True)
    with destination.with_suffix(".log").open("w") as log:
        run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    rows.append(dict(name=name, policy=policy, command=command, exit_code=run.returncode))
    (DOC / "terminal_sweep.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("CANDIDATE_DONE", name, run.returncode, flush=True)
    if run.returncode:
        # Failed multichip execution requires reset before any further probe.
        # Stop the sweep for inspection; recovery is performed by the stage owner.
        break
