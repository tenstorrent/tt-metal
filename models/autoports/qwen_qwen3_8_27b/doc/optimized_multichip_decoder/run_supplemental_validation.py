# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize final advice checks and the post-format installed-default timing."""
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
        "trace_prefill_matrix",
        [
            "python_env/bin/python",
            str(root / "tests/sweep_optimized_multichip_decoder.py"),
            str(doc / "trace_prefill_matrix.json"),
        ],
        env,
    ),
    (
        "supplemental_profile_matrix",
        [
            "python_env/bin/python",
            str(root / "tests/sweep_optimized_multichip_decoder.py"),
            "--profile",
            str(doc / "supplemental_profile_matrix.json"),
        ],
        env,
    ),
    (
        "final_installed_default_matrix",
        [
            "python_env/bin/python",
            str(root / "tests/sweep_optimized_multichip_decoder.py"),
            str(doc / "final_installed_default_matrix.json"),
        ],
        env,
    ),
    (
        "watcher_prefill_trace",
        [
            "bash",
            str(root / "tests/run_optimized_multichip_experiment.sh"),
            "watcher_prefill_trace_stack",
            "--layer",
            "0",
            "--batch",
            "3",
            "--length",
            "33",
            "--stack",
            "--trace-prefill",
            "--repeats",
            "5",
            "--prefill-repeats",
            "10",
        ],
        dict(
            env,
            TT_METAL_WATCHER="10",
            TT_METAL_WATCHER_NOINLINE="1",
            TT_METAL_FABRIC_OPT_LEVEL="O3",
            TT_METAL_CACHE="/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/watcher_tags_fix_stage5",
            TT_METAL_LOGS_PATH=str(doc / "watcher_prefill_trace_stack"),
        ),
    ),
]
for name, command, run_env in steps:
    print("RUN", name, flush=True)
    with (doc / (name + ".log")).open("w") as log:
        result = subprocess.run(command, env=run_env, stdout=log, stderr=subprocess.STDOUT)
    (doc / (name + ".exit_status")).write_text(str(result.returncode) + "\n")
    result.check_returncode()
    print("PASS", name, flush=True)
