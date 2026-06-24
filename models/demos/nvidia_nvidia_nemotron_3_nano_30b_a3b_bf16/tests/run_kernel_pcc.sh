#!/usr/bin/env bash
# Kernel-level HW PCC sweep: runs test_mamba2_ssd_scan_ttlang_hw.py for all
# n_chunks values covering the full ISL range (128 → 262144).
#
# Each n_chunks case opens/closes its own device independently.
# Timeout is extended to 3600s per test to handle large CPU tensor builds
# and PCIe upload times at n_chunks=4096.
set -euo pipefail

TT_METAL_HOME="${TT_METAL_HOME:-/home/ttuser/ssinghal/tt-metal}"
TT_LANG_PYTHON_PATH="${TT_LANG_PYTHON_PATH:-/home/ttuser/ssinghal/tt-lang/build/python_packages}"
PYTHON="${TT_METAL_HOME}/python_env/bin/python"
TEST="${TT_METAL_HOME}/models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/test_mamba2_ssd_scan_ttlang_hw.py"

export TT_METAL_HOME
export TT_LANG_PYTHON_PATH

echo "Resetting device (tt-smi -r all) ..."
tt-smi -r all
sleep 2
echo "  Device reset OK."
echo ""

exec "${PYTHON}" -m pytest "${TEST}" \
    --noconftest -v -s \
    --timeout=3600 \
    "$@"
