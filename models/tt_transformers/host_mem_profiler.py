# profile_tests.py
import os
import subprocess
import sys

import matplotlib.pyplot as plt
from memory_profiler import memory_usage

# def run_test():
#     # Run pytest as a subprocess and capture memory
#     os.environ["MESH_DEVICE"] = "T3K"
#     os.environ["HF_MODEL"] = "Qwen/Qwen2.5-VL-72B-Instruct"
#     os.environ["CI"] = "true"
#     result = subprocess.run(
#         [sys.executable, "-m", "pytest", "models/demos/qwen25_vl/tests/test_model.py", "-v", "-s"], capture_output=False
#     )
#     return result.returncode


# if __name__ == "__main__":
#     mem, retval = memory_usage(
#         (run_test, [], {}),
#         interval=0.1,  # sample every 100ms
#         retval=True,
#         timestamps=True,
#         include_children=True,  # important for pytest subprocesses
#     )
#     print(f"\nPeak memory: {max(m for m, _ in mem):.1f} MiB")
#     print(f"Baseline:    {min(m for m, _ in mem):.1f} MiB")

#     # Unpack (MiB, timestamp) pairs
#     mib_values = [m for m, _ in mem]
#     timestamps = [t for _, t in mem]

#     # Normalize timestamps to start at 0
#     t0 = timestamps[0]
#     elapsed = [t - t0 for t in timestamps]

#     plt.figure(figsize=(12, 5))
#     plt.plot(elapsed, mib_values, "+-k", linewidth=1.5)
#     plt.axhline(max(mib_values), color="red", linestyle="--", linewidth=0.8, label=f"Peak: {max(mib_values):.1f} MiB")
#     plt.axhline(
#         min(mib_values), color="green", linestyle="--", linewidth=0.8, label=f"Baseline: {min(mib_values):.1f} MiB"
#     )
#     plt.xlabel("Time (s)")
#     plt.ylabel("Memory (MiB)")
#     plt.title("Memory Usage Over Time")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("memory_profile.png", dpi=150)
#     #   plt.show()
#     print("Plot saved to memory_profile.png")

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
def plot_results(all_results):
    fig, ax = plt.subplots(figsize=(14, 6))

    for name, elapsed, mib in all_results:
        ax.plot(elapsed, mib, "+-k", linewidth=1.0, markersize=4, label=name)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title("Memory Usage Over Time — Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig("memory_profile_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved → memory_profile_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_results = []

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

        all_results.append((model["name"], elapsed, mib))

    plot_results(all_results)
