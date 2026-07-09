#!/usr/bin/env bash
# Capture attn1 Q/K/V for multiple Stage-2 block depths in ONE denoise run (L4 depth map).
# ltx-perf tip worktree. LTX_TRACED=0 => single eager generation (gen#0): S1(1 step) ->
# upsample -> S2(1 step); the hook captures the CAP_BLOCKS Stage-2 video self-attn triples.
set -uo pipefail
WT=/home/smarton/tt-metal/.claude/worktrees/ltxperf-tip
cd "$WT" || exit 3
export TT_METAL_HOME="$WT"
# opt/l4 on PYTHONPATH so `-p capture_qkv_multi` resolves the plugin from its real location.
export PYTHONPATH="$WT/ttnn:$WT:$WT/tools:$WT/opt/l4"
export GEMMA_PATH="${GEMMA_PATH:-/home/kevinmi/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/68f7ee4fbd59087436ada77ed2d62f373fdd4482/}"
export LTX_CHECKPOINT="${LTX_CHECKPOINT:-/home/kevinmi/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/76730e634e70a28f4e8d51f5e29c08e40e2d8e74/ltx-2.3-22b-distilled-1.1.safetensors}"
# Dedicated ltx-perf-tip caches (match tmp/run_ltx.sh; avoids cross-worktree cache collision).
export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-/home/smarton/.cache/tt-dit-ltxperftip}"
export TT_METAL_CACHE="${TT_METAL_CACHE:-/home/smarton/.cache/tt-metal-cache-ltxperftip}"
export TT_METAL_OPERATION_TIMEOUT_SECONDS="${TT_METAL_OPERATION_TIMEOUT_SECONDS:-300}"
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES="${TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES:-0}"
export LTX_TRACED=0
export LTX_TIME_STAGES=1
export LTX_QUANT="${LTX_QUANT:-all_bf8_lofi}"
export LTX_S1_SIGMAS="${LTX_S1_SIGMAS:-1.0,0.0}"
export LTX_S2_SIGMAS="${LTX_S2_SIGMAS:-0.909375,0.0}"
export NUM_FRAMES="${NUM_FRAMES:-145}"
export HEIGHT="${HEIGHT:-1088}"
export WIDTH="${WIDTH:-1920}"
export NO_PROMPT=1
export SEED="${SEED:-10}"
export RUN_VBENCH=0
export RUN_CLIP=0
export CAP_BLOCKS="${CAP_BLOCKS:-4,8,12,16,20,24,32,40,47}"
export CAP_OUT_DIR="${CAP_OUT_DIR:-$WT/tmp/qkv_depth}"
export CAP_HEADS="${CAP_HEADS:-8}"
export CAP_QNSHARD="${CAP_QNSHARD:-0}"  # 0 => auto (2nd distinct non-cross shard = Stage-2)
echo "=== LTX MULTI-CAPTURE $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) CAP_BLOCKS=$CAP_BLOCKS ==="
echo "env: LTX_QUANT=$LTX_QUANT S1=$LTX_S1_SIGMAS S2=$LTX_S2_SIGMAS frames=$NUM_FRAMES out=$CAP_OUT_DIR"
./python_env/bin/python -m pytest -q -s \
  -p capture_qkv_multi \
  models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled \
  -k bh_4x8sp1tp0_ring -p no:cacheprovider --timeout=0
rc=$?
echo "=== PYTEST RC=$rc ==="
ls -la "$CAP_OUT_DIR" 2>/dev/null
exit $rc
