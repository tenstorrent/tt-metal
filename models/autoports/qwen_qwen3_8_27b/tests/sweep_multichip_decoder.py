# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized coherent whole-decoder policy experiments; stops on failed runtime."""

import json
import subprocess
from pathlib import Path

root = Path("models/autoports/qwen_qwen3_8_27b")
# Pin the recorded control family instead of inheriting later optimized defaults.
base = json.loads((root / "doc/multichip_decoder/sweep_carry_l0.json").read_text())["effective_policy"]
base.update(
    carry_input=False,
    carry_output=False,
    carry_residual=False,
    packed_mlp=False,
    persistent_ccl=False,
    direct_allreduce=False,
)
candidates = {
    "carry": {"carry_input": True, "carry_output": True, "carry_residual": True},
    "block4": {
        "attention_cores": 40,
        "attention_block": 4,
        "gate_cores": 40,
        "gate_block": 4,
        "up_cores": 40,
        "up_block": 4,
    },
    "block8": {
        "attention_cores": 20,
        "attention_block": 8,
        "gate_cores": 20,
        "gate_block": 8,
        "up_cores": 20,
        "up_block": 8,
    },
    "block16": {
        "attention_cores": 10,
        "attention_block": 16,
        "gate_cores": 10,
        "gate_block": 16,
        "up_cores": 10,
        "up_block": 16,
    },
    "minimal": {"minimal_prefill_min": 1},
    "ccl8": {"ccl_dtype": "bfloat8_b"},
    "sharded_ccl8": {"residual_layout": "sharded", "ccl_dtype": "bfloat8_b"},
    "sharded_ring": {"residual_layout": "sharded"},
}
for name, policy in candidates.items():
    policy = {**base, "ring": True, **policy}
    path = root / "doc/multichip_decoder" / f"sweep_{name}_policy.json"
    path.write_text(json.dumps(policy) + "\n")
    command = [
        "bash",
        str(root / "tests/run_multichip_experiment.sh"),
        "sweep_" + name + "_l0",
        "--layer",
        "0",
        "--policy-file",
        str(path),
        "--repeats",
        "15",
    ]
    print("RUN", name, flush=True)
    result = subprocess.run(command)
    if result.returncode:
        raise SystemExit(result.returncode)
    report = json.loads((root / "doc/multichip_decoder" / f"sweep_{name}_l0.json").read_text())
    print("RESULT", name, report["decode_ms"], report["prefill_ms"], report["pcc"], flush=True)
