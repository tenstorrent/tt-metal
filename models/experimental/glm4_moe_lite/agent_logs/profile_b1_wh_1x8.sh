#!/usr/bin/env bash
# Tracy device profile of the WH 1x8 batch-1 baseline (prefill+decode) to find the bottleneck.
# Uses the PCC-passing baseline config (NOT the sweep's BH perf flags, which hang on 1x8).
set -uo pipefail
cd /home/gtobar/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

ISL=${1:-2048}
OUT=/tmp/glm_prof_b1_1x8_isl${ISL}
rm -rf "$OUT"

TT_METAL_GTEST_ETH_DISPATCH=1 \
GLM4_MOE_LITE_ENABLE_MOE=1 GLM4_MOE_LITE_EXPERTS_TT_DTYPE=bf8 GLM4_MOE_LITE_MOE_FP32_ACC=1 \
GLM4_MOE_LITE_TP=1 GLM4_MOE_LITE_ATTN_DP=0 \
GLM4_MOE_LITE_HEAD_PARALLEL_ATTN=0 GLM4_MOE_LITE_HEAD_PARALLEL_KVB2=0 \
GLM4_MOE_LITE_CCL_NUM_LINKS=1 GLM4_MOE_LITE_CCL_TOPOLOGY=ring \
GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0 \
GLM4_MOE_LITE_MAX_PREFILL_CHUNK_SIZE=128 \
GLM4_MOE_LITE_DECODE_MLA_CORE_SCALE=0 GLM4_MOE_LITE_DECODE_L1_ACT=1 GLM4_MOE_LITE_EP_L1=1 \
GLM4_MOE_LITE_SIGNPOST=1 \
python -m tracy -r -p -o "$OUT" -- \
  models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt Summarize --simulate-context-len "$ISL" --min-cache-tokens $((ISL+64)) \
  --max-new-tokens 4 --batch-size 1 \
  --mesh-rows 1 --mesh-cols 8 --kv-cache-dtype bf16 \
  --phase both --cache-dir "$HOME/.cache/ttnn/models/glm4_moe_lite/wh_1x8" 2>&1

echo "=== RUN EXIT: $? ==="
find "$OUT" -name 'ops_perf_results_*.csv' 2>/dev/null
echo "=== DONE_PROF ==="
