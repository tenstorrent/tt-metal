# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize final validation; any failed child stops subsequent device work."""
import os
import subprocess
from pathlib import Path

root = Path("models/autoports/qwen_qwen3_8_27b")
doc = root / "doc/optimized_multichip_decoder"
env = dict(
    os.environ, TT_METAL_CACHE="/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/optimized_multichip_tags_stage5"
)
steps = [
    (
        "final_stress_matrix",
        [
            "python_env/bin/python",
            str(root / "tests/sweep_optimized_multichip_decoder.py"),
            str(doc / "final_stress_matrix.json"),
        ],
    ),
    ("final_watcher", ["python_env/bin/python", str(root / "tests/watch_multichip_decoder.py")]),
    (
        "final_capacity_matrix",
        [
            "python_env/bin/python",
            str(root / "tests/sweep_optimized_multichip_decoder.py"),
            str(doc / "final_capacity_matrix.json"),
        ],
    ),
    (
        "memory_plan_validated",
        [
            "python_env/bin/python",
            str(root / "tests/multichip_memory_plan.py"),
            "--stage",
            "optimized_multichip_decoder",
            "--policy-file",
            str(doc / "final_default_policy.json"),
        ],
    ),
    (
        "final_profile_matrix",
        [
            "python_env/bin/python",
            str(root / "tests/sweep_optimized_multichip_decoder.py"),
            "--profile",
            str(doc / "final_profile_matrix.json"),
        ],
    ),
    (
        "final_tail_timing_matrix",
        [
            "python_env/bin/python",
            str(root / "tests/sweep_optimized_multichip_decoder.py"),
            str(doc / "final_tail_timing_matrix.json"),
        ],
    ),
]
for name, command in steps:
    print("RUN", name, flush=True)
    with (doc / (name + ".log")).open("w") as log:
        result = subprocess.run(
            command,
            env=dict(env, QWEN_MULTICHIP_STAGE="optimized_multichip_decoder"),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    (doc / (name + ".exit_status")).write_text(str(result.returncode) + "\n")
    result.check_returncode()
    print("PASS", name, flush=True)
