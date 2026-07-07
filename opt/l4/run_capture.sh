#!/usr/bin/env bash
# Capture ONE real post-RoPE attn1 Q/K/V from a live LTX Stage-2 denoise step.
# Runs the real distilled pipeline UNTRACED (LTX_TRACED=0) for 1 S1 step + 1 S2 step,
# with tmp/capture_qkv_plugin.py monkeypatching ring-SDPA to dump block CAP_BLOCK.
set -uo pipefail
WT=/home/smarton/tt-metal/.claude/worktrees/ltx-perf-clean
cd "$WT" || exit 3

export TT_METAL_HOME="$WT"
# add tmp/ so `-p capture_qkv_plugin` is importable
export PYTHONPATH="$WT/ttnn:$WT:$WT/tools:$WT/tmp"
export GEMMA_PATH="${GEMMA_PATH:-/home/kevinmi/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/68f7ee4fbd59087436ada77ed2d62f373fdd4482/}"
export LTX_CHECKPOINT="${LTX_CHECKPOINT:-/home/kevinmi/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/76730e634e70a28f4e8d51f5e29c08e40e2d8e74/ltx-2.3-22b-distilled-1.1.safetensors}"
export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-/home/smarton/.cache/tt-dit}"
export TT_METAL_CACHE="${TT_METAL_CACHE:-/home/smarton/.cache/tt-metal-cache-ltxperf650}"
export TT_METAL_OPERATION_TIMEOUT_SECONDS="${TT_METAL_OPERATION_TIMEOUT_SECONDS:-300}"
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES="${TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES:-0}"

# Realistic fast-mode activations, but UNTRACED so the python monkeypatch runs and
# to_torch is legal (no trace capture). Minimal sigmas: 1 S1 step + 1 S2 step.
export LTX_FAST=1
export LTX_TRACED=0
export LTX_QUANT="${LTX_QUANT:-all_bf8_lofi}"
export LTX_S1_SIGMAS="${LTX_S1_SIGMAS:-1.0,0.0}"
export LTX_S2_SIGMAS="${LTX_S2_SIGMAS:-0.909375,0.0}"
export LTX_TIME_STAGES=1
export NUM_FRAMES="${NUM_FRAMES:-145}"
export NO_PROMPT=1
export SEED="${SEED:-10}"
export RUN_VBENCH=0
export RUN_CLIP=0

export CAP_BLOCK="${CAP_BLOCK:-24}"
export CAP_OUT="${CAP_OUT:-$WT/tmp/real_attn_qkv.pt}"
export CAP_HEADS="${CAP_HEADS:-8}"

echo "=== LTX CAPTURE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "buildid: $(cd "$WT" && git rev-parse --short HEAD)"
echo "env: LTX_TRACED=$LTX_TRACED S1=$LTX_S1_SIGMAS S2=$LTX_S2_SIGMAS CAP_BLOCK=$CAP_BLOCK CAP_OUT=$CAP_OUT CAP_HEADS=$CAP_HEADS"

./python_env/bin/python -m pytest -q -s \
  -p capture_qkv_plugin \
  models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled \
  -k bh_4x8sp1tp0_ring -p no:cacheprovider --timeout=0
rc=$?
echo "=== PYTEST RC=$rc ==="
echo "=== capture file: ==="
ls -la "$CAP_OUT" 2>/dev/null || echo "NO CAPTURE FILE"
exit $rc
