# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize named real-activation decoder experiments and retain failed candidates."""

import argparse
import json
import subprocess
from pathlib import Path

BASE = dict(
    dram=True, sharded_norm=True, carry_residual=True, attention_block=4, gate_block=4, up_block=4, down_block=17
)
for role in ("attention", "gate", "up", "down"):
    BASE[role + "_dtype"] = "bfloat4_b"
    BASE[role + "_fidelity"] = "LoFi"

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, required=True)
    a = p.parse_args()
    for row in json.loads(a.matrix.read_text()):
        # Historical matrices are deltas from BASE. Explicit default runs must
        # reach the decoder's default constructor rather than that old baseline.
        policy = {} if row.get("default_policy", False) else BASE | row.get("policy", {})
        cmd = [
            "bash",
            "models/autoports/qwen_qwen3_8_27b/tests/run_optimization_experiment.sh",
            row["name"],
            "--layer",
            str(row.get("layer", 3)),
            "--length",
            str(row.get("length", 128)),
            "--batch",
            str(row.get("batch", 1)),
            "--benchmark",
            "--activations",
            "/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/optimized_decoder_activations",
            "--policy",
            json.dumps(policy),
        ] + row.get("args", [])
        print("START", row["name"], flush=True)
        result = subprocess.run(cmd)
        print("RESULT", row["name"], result.returncode, flush=True)
        if result.returncode in (124, 137):
            raise RuntimeError("Device job timed out; stop sweep for triage/recovery")
