#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <kernelbench-args...>"
    echo "Example: $0 --measure-device-kernel --regimes decode_1d_dram_bound_small_m"
    exit 1
fi

source /home/ubuntu/tt-metal/python_env/bin/activate

# Device profiler path requires a Tracy-enabled build.
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_MID_RUN_DUMP=1
export TT_METAL_PROFILER_CPP_POST_PROCESS=1

ARCH_NAME=wormhole_b0 python3 \
    /home/ubuntu/tt-metal/tests/sweep_framework/benchmark_protocol/matmul_n150_kernelbench.py "$@"
