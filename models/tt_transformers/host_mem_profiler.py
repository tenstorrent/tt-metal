# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger
from memory_profiler import memory_usage

PYTEST_EXTRA_ARGS = ["-v", "-s"]
OUTPUT_ROOT = Path("profiling_results")


# ── Runner ────────────────────────────────────────────────────────────────────
def make_runner(model_cfg):
    """Return a zero-arg callable that runs a model's test suite."""

    def run():
        env = os.environ.copy()
        env.update(model_cfg["env"])
        result = subprocess.run(
            [sys.executable, "-m", "pytest", model_cfg["test"], *PYTEST_EXTRA_ARGS],
            capture_output=False,
            env=env,
        )
        return result.returncode

    return run


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_results(name, elapsed, mib, out_dir):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(elapsed, mib, "+-k", linewidth=1.0, markersize=4, label=name)
    ax.axhline(max(mib), color="red", linestyle="--", linewidth=0.8, label=f"Peak: {max(mib):.1f} MiB")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title("Memory Usage Over Time — Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "memory_profile_comparison.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved at {os.path.join(out_dir, 'memory_profile_comparison.png')}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile host-side memory usage of a model test")
    parser.add_argument("--name", required=True, help="Model display name (e.g. Llama-3.1-8B)")
    parser.add_argument("--mesh-device", required=True, help="MESH_DEVICE value (e.g. N150, T3K)")
    parser.add_argument("--hf-model", required=True, help="Hugging Face model ID")
    parser.add_argument("--test", required=True, help="Pytest target path")
    args = parser.parse_args()

    model = {
        "name": args.name,
        "env": {
            "MESH_DEVICE": args.mesh_device,
            "HF_MODEL": args.hf_model,
        },
        "test": args.test,
    }

    logger.info(f"\n{'='*60}\nProfiling: {model['name']}\n{'='*60}")

    mem_ts, _ = memory_usage(
        (make_runner(model), [], {}),
        interval=0.1,
        retval=True,
        timestamps=True,
        include_children=True,
    )

    mib = [m for m, _ in mem_ts]
    ts = [t for _, t in mem_ts]
    t0 = ts[0]
    elapsed = [t - t0 for t in ts]

    logger.info(f"  Peak:     {max(mib):.1f} MiB")
    logger.info(f"  Baseline: {min(mib):.1f} MiB")

    safe_name = args.name.replace("/", "_").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(os.path.abspath(os.path.join(OUTPUT_ROOT, f"{safe_name}_{timestamp}")))
    if out_dir.is_relative_to(OUTPUT_ROOT.resolve()):
        out_dir.mkdir(parents=True, exist_ok=True)

    plot_results(model["name"], elapsed, mib, out_dir)
