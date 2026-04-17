# profile_tests.py
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from memory_profiler import memory_usage

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry: display name, env vars, pytest target
MODELS = [
    {
        "name": "Llama-3.1-8B",
        "env": {
            "MESH_DEVICE": "N150",
            "HF_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
        },
        "test": "models/tt_transformers/tests/test_mlp.py",
    },
    {
        "name": "Llama-3.1-70B",
        "env": {
            "MESH_DEVICE": "T3K",
            "HF_MODEL": "meta-llama/Llama-3.1-70B-Instruct",
        },
        "test": "models/tt_transformers/tests/test_mlp.py",
    },
    # {
    #     "name": "Qwen2.5-VL-72B",
    #     "env": {
    #         "MESH_DEVICE": "T3K",
    #         "HF_MODEL": "Qwen/Qwen2.5-VL-72B-Instruct",
    #     },
    #     "test": "models/demos/qwen25_vl/tests/test_mlp.py",
    # },
    # {
    #     "name": "Falcon-7B",
    #     "env": {
    #         "MESH_DEVICE": "N150",
    #         "HF_MODEL": "tiiuae/falcon-7b-instruct",
    #     },
    #     "test": "models/demos/ttnn_falcon7b/tests/test_falcon_mlp.py",
    # },
]

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

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title("Memory Usage Over Time — Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "memory_profile_comparison.png"), dpi=150)
    plt.close(fig)
    print("Saved → memory_profile_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Profiling: {model['name']}")
        print(f"{'='*60}")

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

        print(f"  Peak:     {max(mib):.1f} MiB")
        print(f"  Baseline: {min(mib):.1f} MiB")

        name = model["name"]
        safe_name = name.replace("/", "_").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_ROOT / f"{safe_name}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_results(model["name"], elapsed, mib, out_dir)
