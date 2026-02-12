#!/usr/bin/env python3
"""
Collect device performance baselines for all DeepSeek V3 fused ops.
Runs each test twice:
  - Prefill seq_len=128 with program cache (eager)
  - Decode seq_len=1 with program cache (trace)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

# Map of op names to their test paths
OPS_MAP = {
    "embedding": "models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_embedding.py::test_ds_embedding_device_perf",
    "all_gather_embedding": "models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_all_gather_embedding.py::test_ds_all_gather_embedding_device_perf",
    "lm_head": "models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_lm_head.py::test_ds_lm_head_device_perf",
    "rms_norm": "models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_rms_norm.py::test_ds_rms_norm_device_perf",
    "distributed_norm": "models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_distributed_norm.py::test_ds_distributed_norm_device_perf",
    "ff1_3": "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff1_3.py::test_ds_ff1_3_device_perf",
    "all_gather_preff1_3": "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_all_gather_preff1_3.py::test_ds_all_gather_preff1_3_device_perf",
    "mul": "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_mul.py::test_ds_mul_device_perf",
    "ff2": "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff2.py::test_ds_ff2_device_perf",
    "reduce_scatter": "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_reduce_scatter_post_ff2.py::test_ds_reduce_scatter_post_ff2_device_perf",
}


def extract_perf_metrics(output: str) -> tuple[float, float]:
    """Extract kernel and op_to_op times from test output."""
    match = re.search(r"Device perf totals: kernel=([0-9.]+) us, op_to_op=([0-9.]+) us", output)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def run_device_perf_test(test_path: str, k_filter: str, op_name: str, config_name: str) -> tuple[float, float]:
    """Run a single device perf test and extract metrics."""
    print(f"\n{'='*60}")
    print(f"Running: {op_name} - {config_name}")
    print(f"{'='*60}")

    # Clean profiler output
    profiler_dir = Path("generated/profiler/deepseek_v3_fused_ops_device_perf")
    if profiler_dir.exists():
        subprocess.run(["rm", "-rf", str(profiler_dir)], check=True)

    # Run test
    cmd = ["pytest", test_path, "-k", k_filter, "--timeout", "600", "-v", "-s"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract metrics
    kernel_us, op_to_op_us = extract_perf_metrics(result.stdout + result.stderr)

    if kernel_us is not None:
        print(f"✓ {config_name}: kernel={kernel_us:.3f}us, op_to_op={op_to_op_us:.3f}us")
        return kernel_us, op_to_op_us
    else:
        print(f"✗ Failed to extract metrics from output")
        print(f"Return code: {result.returncode}")
        # Print last 50 lines of output for debugging
        lines = (result.stdout + result.stderr).split("\n")
        print("\n".join(lines[-50:]))
        return None, None


def main():
    baselines = {}

    for op_name, test_path in OPS_MAP.items():
        print(f"\n{'#'*60}")
        print(f"# OP: {op_name}")
        print(f"{'#'*60}")

        op_baselines = {}

        # Prefill seq_len=128 (eager + program_cache)
        kernel, op_to_op = run_device_perf_test(test_path, "prefill and 128", op_name, "prefill_128_eager_pcache")
        if kernel is not None:
            op_baselines["prefill_128"] = {"kernel_us": kernel, "op_to_op_us": op_to_op}

        # Decode seq_len=1 (trace + program_cache)
        kernel, op_to_op = run_device_perf_test(test_path, "decode and 1", op_name, "decode_1_trace_pcache")
        if kernel is not None:
            op_baselines["decode_1"] = {"kernel_us": kernel, "op_to_op_us": op_to_op}

        baselines[op_name] = op_baselines

    # Save baselines to JSON
    output_file = "device_perf_baselines.json"
    with open(output_file, "w") as f:
        json.dump(baselines, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All Device Perf Tests Complete!")
    print(f"Baselines saved to: {output_file}")
    print(f"{'='*60}\n")

    # Print summary
    print(json.dumps(baselines, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
