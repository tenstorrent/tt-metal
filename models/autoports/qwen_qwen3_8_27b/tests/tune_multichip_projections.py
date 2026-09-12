# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision-locked projection and collective closure on the selected TP4 family."""

import json
import subprocess
import sys
from pathlib import Path

root = Path("models/autoports/qwen_qwen3_8_27b")
base = json.loads((root / "doc/multichip_decoder/tune_control_l0.json").read_text())["effective_policy"]
base.update(packed_mlp=False, persistent_ccl=False, direct_allreduce=False)
base.update(
    {
        "ring": True,
        "attention_cores": 10,
        "attention_block": 16,
        "gate_cores": 10,
        "gate_block": 16,
        "up_cores": 10,
        "up_block": 16,
        "carry_input": True,
        "carry_output": True,
        "carry_residual": True,
    }
)
candidates = {
    "control": {},
    "output4": {"output_cores": 12, "output_block": 4},
    "output6": {"output_cores": 8, "output_block": 6},
    "output12": {"output_cores": 4, "output_block": 12},
    "down34": {"down_cores": 4, "down_block": 34},
    "block32": {
        "attention_cores": 5,
        "attention_block": 32,
        "gate_cores": 5,
        "gate_block": 32,
        "up_cores": 5,
        "up_block": 32,
    },
    "packed": {"packed_mlp": True},
    "ccl8": {"ccl_dtype": "bfloat8_b"},
    "persistent": {"persistent_ccl": True},
    "links2": {"num_links": 2},
    "hifi2": {role + "_fidelity": "HiFi2" for role in ("attention", "output", "gate", "up", "down")},
    "epilogue": {"gate_epilogue": True},
}
selected = sys.argv[1:] or list(candidates)
for name in selected:
    policy = candidates[name]
    policy = {**base, **policy}
    path = root / "doc/multichip_decoder" / f"tune_{name}_policy.json"
    path.write_text(json.dumps(policy) + "\n")
    command = [
        "bash",
        str(root / "tests/run_multichip_experiment.sh"),
        "tune_" + name + "_l0",
        "--layer",
        "0",
        "--policy-file",
        str(path),
        "--repeats",
        "20",
    ]
    print("RUN", name, flush=True)
    result = subprocess.run(command)
    if result.returncode:
        raise SystemExit(result.returncode)
    report = json.loads((root / "doc/multichip_decoder" / f"tune_{name}_l0.json").read_text())
    print("RESULT", name, report["decode_ms"], report["prefill_ms"], report["pcc"], flush=True)
