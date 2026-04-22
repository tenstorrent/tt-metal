# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import argparse
import importlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def _require(package, import_name=None):
    import_name = import_name or package
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"Installing missing dependency: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        importlib.invalidate_caches()  # flush finder cache so the new package is visible


_require("memory-profiler", "memory_profiler")
_require("matplotlib")

import matplotlib.pyplot as plt
from memory_profiler import memory_usage

PYTEST_EXTRA_ARGS = ["-v", "-s"]
OUTPUT_ROOT = Path("profiling_results")


# ── Runner ────────────────────────────────────────────────────────────────────
def make_runner(model_cfg, k_filter=None):
    """Return a zero-arg callable that runs a model's test suite."""

    def run():
        env = os.environ.copy()
        env.update(model_cfg["env"])
        cmd = [sys.executable, "-m", "pytest", model_cfg["test"], *PYTEST_EXTRA_ARGS]
        if k_filter:
            cmd += ["-k", k_filter]
        result = subprocess.run(cmd, capture_output=False, env=env)
        return result.returncode

    return run


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_results(name, elapsed, mib, out_dir):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(elapsed, mib, "+-k", linewidth=1.0, markersize=4, label=name)
    ax.axhline(max(mib), color="red", linestyle="--", linewidth=0.8, label=f"Peak: {max(mib):.1f} MiB")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title("Memory Usage Over Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "memory_profile.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved at {os.path.join(out_dir, 'memory_profile.png')}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile host-side memory usage of a model test")
    parser.add_argument("--name", default="Llama-3.2-1B-Instruct", help="Model display name (e.g. Llama-3.1-8B)")
    parser.add_argument("--mesh-device", default="N150", help="MESH_DEVICE value (e.g. N150, T3K)")
    parser.add_argument("--hf-model", default="meta-llama/Llama-3.2-1B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--test", default="models/tt_transformers/demo/simple_text_demo.py", help="Pytest target path")
    parser.add_argument("-k", default=None, help="pytest -k filter expression")
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

    mem_ts, returncode = memory_usage(
        (make_runner(model, args.k), [], {}),
        interval=0.1,
        retval=True,
        timestamps=True,
        include_children=True,
    )
    if returncode:
        logger.error(f"pytest exited with code {returncode} — skipping plot")
        sys.exit(returncode)

    mib = [m for m, _ in mem_ts]
    ts = [t for _, t in mem_ts]
    t0 = ts[0]
    elapsed = [t - t0 for t in ts]

    logger.info(f"  Peak:     {max(mib):.1f} MiB")
    logger.info(f"  Baseline: {min(mib):.1f} MiB")

    safe_name = args.name.replace("/", "_").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = OUTPUT_ROOT.resolve()
    out_dir = (output_root / f"{safe_name}_{timestamp}").resolve()
    if not out_dir.is_relative_to(output_root):
        raise ValueError(f"Refusing to write outside output root: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_results(model["name"], elapsed, mib, out_dir)
