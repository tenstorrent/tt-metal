# profile_tests.py
import os
import subprocess
import sys

import matplotlib.pyplot as plt
from memory_profiler import memory_usage


def run_test():
    # Run pytest as a subprocess and capture memory
    os.environ["MESH_DEVICE"] = "T3K"
    os.environ["HF_MODEL"] = "Qwen/Qwen2.5-VL-72B-Instruct"
    os.environ["CI"] = "true"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "models/demos/qwen25_vl/tests/test_model.py", "-v", "-s"], capture_output=False
    )
    return result.returncode


if __name__ == "__main__":
    mem, retval = memory_usage(
        (run_test, [], {}),
        interval=0.1,  # sample every 100ms
        retval=True,
        timestamps=True,
        include_children=True,  # important for pytest subprocesses
    )
    print(f"\nPeak memory: {max(m for m, _ in mem):.1f} MiB")
    print(f"Baseline:    {min(m for m, _ in mem):.1f} MiB")

    # Unpack (MiB, timestamp) pairs
    mib_values = [m for m, _ in mem]
    timestamps = [t for _, t in mem]

    # Normalize timestamps to start at 0
    t0 = timestamps[0]
    elapsed = [t - t0 for t in timestamps]

    plt.figure(figsize=(12, 5))
    plt.plot(elapsed, mib_values, "+-k", linewidth=1.5)
    plt.axhline(max(mib_values), color="red", linestyle="--", linewidth=0.8, label=f"Peak: {max(mib_values):.1f} MiB")
    plt.axhline(
        min(mib_values), color="green", linestyle="--", linewidth=0.8, label=f"Baseline: {min(mib_values):.1f} MiB"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Memory (MiB)")
    plt.title("Memory Usage Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("memory_profile.png", dpi=150)
    #   plt.show()
    print("Plot saved to memory_profile.png")
