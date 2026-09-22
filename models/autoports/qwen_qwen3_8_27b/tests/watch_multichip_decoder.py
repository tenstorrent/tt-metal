# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run source-built watcher separately from every profiler command."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = os.getenv("QWEN_MULTICHIP_STAGE", "multichip_decoder")
DOC = ROOT / "doc" / STAGE
WRAPPER = (
    "run_optimized_multichip_experiment.sh" if STAGE == "optimized_multichip_decoder" else "run_multichip_experiment.sh"
)
CASES = {
    "watcher_linear_tail": ["--layer", "0", "--length", "2049"],
    "watcher_linear_batch32": ["--layer", "0", "--batch", "32", "--length", "257"],
    "watcher_full_batch32": ["--layer", "3", "--batch", "32", "--length", "257"],
    "watcher_stack_batch3": ["--layer", "0", "--batch", "3", "--length", "33", "--stack"],
}
for name, options in CASES.items():
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("TT_METAL_PROFILER") or key == "TT_METAL_DEVICE_PROFILER":
            del env[key]
    for key in tuple(env):
        if key.startswith("TT_METAL_WATCHER_DISABLE"):
            del env[key]
    env.update(
        TT_METAL_WATCHER="10",
        TT_METAL_WATCHER_NOINLINE="1",
        TT_METAL_FABRIC_OPT_LEVEL="O3",
        TT_METAL_CACHE=str(ROOT.parents[2] / "../tt-metal-cache/watcher_tags_fix_stage5"),
        TT_METAL_LOGS_PATH=str(DOC / name),
    )
    subprocess.run(
        ["bash", str(ROOT / "tests" / WRAPPER), name, *options, "--repeats", "30"],
        env=env,
        check=True,
    )
    print(name, "PASS", flush=True)
