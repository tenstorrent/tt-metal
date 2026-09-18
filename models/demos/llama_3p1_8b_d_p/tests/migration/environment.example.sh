#!/usr/bin/env bash
# Copy outside the checkout, set explicit local paths, and hash that file in the plan.
# This environment does not reserve or reset hardware and does not arm a plan.
export PREFILL_REPO=/configure/tt-metal
export PREFILL_PYTHON=/configure/python-env/bin/python
export HF_MODEL=/configure/Llama-3.1-8B-Instruct
export PREFILL_HF_MODEL="$HF_MODEL" LLAMA31_8B_HF_MODEL="$HF_MODEL"
export TT_METAL_HOME="$PREFILL_REPO"
export PYTHONPATH="$PREFILL_REPO:$PREFILL_REPO/ttnn"
export LD_LIBRARY_PATH="$PREFILL_REPO/build_Release/lib:/configure/native-dependencies/lib"
export PREFILL_FABRIC_MODE=1d_ring
unset TT_METAL_SLOW_DISPATCH_MODE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Set TTNN_CONFIG_OVERRIDES with absolute cache/model_cache/tmp/report paths within
# the source tree's shared parent. verify_native_env.py checks their resolved origin.
