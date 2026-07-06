#!/usr/bin/env bash
set -u
cd "$(git rev-parse --show-toplevel)"
source python_env/bin/activate
CACHE="$HOME/.cache/ttnn/models/glm4_moe_lite/wh_1x8"
SCRIPT="models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py"
ISL=${ISL:-512}
export GLM4_MOE_LITE_ENABLE_MOE=1 GLM4_MOE_LITE_EXPERTS_TT_DTYPE=bf8 GLM4_MOE_LITE_MOE_FP32_ACC=1
export GLM4_MOE_LITE_TP=1 GLM4_MOE_LITE_ATTN_DP=0
export GLM4_MOE_LITE_HEAD_PARALLEL_ATTN=0 GLM4_MOE_LITE_HEAD_PARALLEL_KVB2=0
export GLM4_MOE_LITE_CCL_NUM_LINKS=1 GLM4_MOE_LITE_CCL_TOPOLOGY=ring
export GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0 TT_METAL_GTEST_ETH_DISPATCH=1
export GLM4_MOE_LITE_MAX_PREFILL_CHUNK_SIZE=128
export GLM4_MOE_LITE_DECODE_MLA_CORE_SCALE=0 GLM4_MOE_LITE_DECODE_L1_ACT=1 GLM4_MOE_LITE_EP_L1=1
export GLM4_MOE_LITE_PREFILL_TRACE=${GLM4_MOE_LITE_PREFILL_TRACE:-1}
timeout 900 python "$SCRIPT" \
  --prompt "Summarize the following document. " \
  --simulate-context-len "$ISL" --min-cache-tokens $((ISL + 16)) \
  --max-new-tokens 4 --batch-size 1 --mesh-rows 1 --mesh-cols 8 \
  --kv-cache-dtype bf16 --phase prefill --warmup \
  --cache-dir "$CACHE" 2>&1
echo "=== SMOKE EXIT $? ==="
