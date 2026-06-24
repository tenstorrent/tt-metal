#!/usr/bin/env bash
# ISL sweep: prefill TTFT, tok/s, and PCC for vanilla vs tt-lang SSD path.
set -euo pipefail

TT_METAL_HOME="${TT_METAL_HOME:-/home/ttuser/ssinghal/tt-metal}"
TT_LANG_PYTHON_PATH="${TT_LANG_PYTHON_PATH:-/home/ttuser/ssinghal/tt-lang/build/python_packages}"
PYTHON="${TT_METAL_HOME}/python_env/bin/python"

export TT_METAL_HOME
export TT_LANG_PYTHON_PATH

exec "${PYTHON}" \
  "${TT_METAL_HOME}/models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/bench_prefill_isl_sweep.py" \
  "$@"
