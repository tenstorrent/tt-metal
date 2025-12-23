#!/usr/bin/env python3
"""
OpenVLA TTNN Demo - Runs the model test directly.

Usage:
    export OPENVLA_WEIGHTS=<path_to_openvla_weights>
    export HF_MODEL=meta-llama/Llama-2-7b-hf
    cd /path/to/tt-metal

    # Quick test (5 iterations)
    python models/experimental/openvla/demo/demo.py

    # Performance benchmark (100 iterations)
    python models/experimental/openvla/demo/demo.py --benchmark
"""

import os
import sys
import argparse

# Ensure we're in the right directory
if not os.path.exists("models/experimental/openvla"):
    print("❌ Please run from tt-metal root directory")
    sys.exit(1)

# Parse arguments
parser = argparse.ArgumentParser(description="OpenVLA TTNN Demo")
parser.add_argument("--benchmark", action="store_true", help="Run 100 iterations for performance benchmarking")
parser.add_argument(
    "--iterations", type=int, default=None, help="Number of iterations (default: 5, or 100 with --benchmark)"
)
args = parser.parse_args()

# Check required env vars
if not os.environ.get("OPENVLA_WEIGHTS"):
    print("❌ OPENVLA_WEIGHTS not set")
    print("   export OPENVLA_WEIGHTS=<path_to_weights>")
    sys.exit(1)

if not os.environ.get("HF_MODEL"):
    print("❌ HF_MODEL not set")
    print("   export HF_MODEL=meta-llama/Llama-2-7b-hf")
    sys.exit(1)

# Determine iterations
if args.iterations:
    iterations = args.iterations
elif args.benchmark:
    iterations = 100
else:
    iterations = 5

print("=" * 60)
print(f"OpenVLA Demo - Running {iterations} iterations")
print("=" * 60)

# Override iterations in pytest via environment variable
os.environ["OPENVLA_TEST_ITERATIONS"] = str(iterations)

# Run pytest on the test function
import subprocess

result = subprocess.run(
    [sys.executable, "-m", "pytest", "models/experimental/openvla/tt/open_vla.py::test_openvla_model", "-v", "-s"],
    env=os.environ,
)
sys.exit(result.returncode)
