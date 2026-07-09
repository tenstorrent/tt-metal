#!/usr/bin/env bash
# LTX e2e runner for the 3s opt loop. Reproducible env bundle (ltxperf650 recipe +
# the pinned-mem trace fix). Overridable via exported vars before calling:
#   LTX_FAST, LTX_QUANT, LTX_S1_SIGMAS, LTX_S2_SIGMAS, NUM_FRAMES, RUN_VBENCH,
#   RUN_CLIP, OUTPUT_PATH, SEED, plus any experimental flag a worker is testing.
set -uo pipefail
WT=/home/smarton/tt-metal/.claude/worktrees/ltx-perf-clean
cd "$WT" || exit 3

export TT_METAL_HOME="$WT"
export PYTHONPATH="$WT/ttnn:$WT:$WT/tools"
export GEMMA_PATH="${GEMMA_PATH:-/home/kevinmi/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/68f7ee4fbd59087436ada77ed2d62f373fdd4482/}"
export LTX_CHECKPOINT="${LTX_CHECKPOINT:-/home/kevinmi/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/76730e634e70a28f4e8d51f5e29c08e40e2d8e74/ltx-2.3-22b-distilled-1.1.safetensors}"
export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-/home/smarton/.cache/tt-dit}"
export TT_METAL_CACHE="${TT_METAL_CACHE:-/home/smarton/.cache/tt-metal-cache-ltxperf650}"
export TT_METAL_OPERATION_TIMEOUT_SECONDS="${TT_METAL_OPERATION_TIMEOUT_SECONDS:-300}"
# Required on 4x8 or trace capture hangs at end_trace_capture (handoff, Kevin's fix).
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES="${TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES:-0}"

export LTX_TIME_STAGES=1
export LTX_TRACED="${LTX_TRACED:-1}"
export NUM_FRAMES="${NUM_FRAMES:-145}"
export NO_PROMPT="${NO_PROMPT:-1}"
export SEED="${SEED:-10}"
export RUN_VBENCH="${RUN_VBENCH:-0}"
export RUN_CLIP="${RUN_CLIP:-0}"

echo "=== LTX RUN $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "buildid: $(cd "$WT" && git rev-parse --short HEAD)  bin=$(md5sum build_Release/lib/libtt_metal.so 2>/dev/null | cut -c1-12)"
echo "env: LTX_FAST=${LTX_FAST:-0} LTX_QUANT=${LTX_QUANT:-unset} S1=${LTX_S1_SIGMAS:-default} S2=${LTX_S2_SIGMAS:-default} frames=$NUM_FRAMES vbench=$RUN_VBENCH clip=$RUN_CLIP"

./python_env/bin/python -m pytest -q -s \
  models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled \
  -k bh_4x8sp1tp0_ring -p no:cacheprovider --timeout=0
rc=$?
echo "=== PYTEST RC=$rc ==="
exit $rc
