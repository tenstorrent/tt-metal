# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refresh final evidence after the review's redundant-cast repair."""

import os
import subprocess
from pathlib import Path

root = Path("models/autoports/qwen_qwen3_8_27b")
doc = root / "doc/optimized_multichip_decoder"
env = dict(
    os.environ,
    TT_METAL_CACHE="/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/optimized_multichip_tags_stage5",
    QWEN_MULTICHIP_STAGE="optimized_multichip_decoder",
    QWEN_REUSE_OPTIMIZED_BASELINE="1",
    PYTHONPATH=".",
)
sweep = ["python_env/bin/python", str(root / "tests/sweep_optimized_multichip_decoder.py")]
steps = [
    (
        "final_correctness_pytest",
        ["python_env/bin/python", "-m", "pytest", str(root / "tests/test_multichip_decoder.py"), "-x", "-v"],
    ),
    ("review_trace_prefill_matrix", [*sweep, str(doc / "review_trace_prefill_matrix.json")]),
    ("review_default_matrix", [*sweep, str(doc / "review_final_installed_default_matrix.json")]),
    ("final_stress_matrix", [*sweep, str(doc / "final_stress_matrix.json")]),
    ("final_watcher", ["python_env/bin/python", str(root / "tests/watch_multichip_decoder.py")]),
    (
        "watcher_review_prefill_trace_stack",
        [
            "bash",
            str(root / "tests/run_optimized_multichip_experiment.sh"),
            "watcher_review_prefill_trace_stack",
            "--layer",
            "0",
            "--length",
            "128",
            "--stack",
            "--trace-prefill",
            "--repeats",
            "5",
            "--prefill-repeats",
            "10",
        ],
    ),
    ("final_capacity_matrix", [*sweep, str(doc / "final_capacity_matrix.json")]),
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
    ("review_profile_matrix", [*sweep, "--profile", str(doc / "review_profile_matrix.json")]),
    ("final_tail_timing_matrix", [*sweep, str(doc / "final_tail_timing_matrix.json")]),
]
for name, command in steps:
    print("RUN", name, flush=True)
    run_env = env
    if name == "watcher_review_prefill_trace_stack":
        run_env = {key: value for key, value in env.items() if not key.startswith("TT_METAL_WATCHER_DISABLE")}
        run_env.update(
            TT_METAL_WATCHER="10",
            TT_METAL_WATCHER_NOINLINE="1",
            TT_METAL_FABRIC_OPT_LEVEL="O3",
            TT_METAL_CACHE="/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/watcher_tags_fix_stage5",
            TT_METAL_LOGS_PATH=str(doc / name),
        )
    with (doc / (name + ".log")).open("w") as log:
        result = subprocess.run(command, env=run_env, stdout=log, stderr=subprocess.STDOUT)
    (doc / (name + ".exit_status")).write_text(str(result.returncode) + "\n")
    result.check_returncode()
    print("PASS", name, flush=True)
