#!/usr/bin/env bash
# Extend the prefill_pcm A/B (PCC + timing + chunk decision) to batches 8/16/32 at ISL-128.
# Each batch: OLD 4-chunk vs NEW adaptive, full-depth logits PCC + median prefill time.
set -uo pipefail
cd /home/gtobar/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

COMMON_ENV=(
  TT_METAL_GTEST_ETH_DISPATCH=1
  GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0
  GLM4_MOE_LITE_CCL_NUM_LINKS=1
  GLM4_MOE_LITE_CCL_TOPOLOGY=linear
  GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 GLM4_MOE_LITE_FUSE_QKV_A=1
  GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1
  GLM4_MOE_LITE_DECODE_L1_ACT=1 GLM4_MOE_LITE_EP_L1=1
  GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1 GLM4_MOE_LITE_SKIP_TYPECAST=1
  GLM4_MOE_LITE_MOE_CHUNK_DEBUG=1
)

run_batch () {
  local B=$1 KVDT=$2 BATCHED=$3
  echo "############################## BATCH=$B kv=$KVDT batched_prefill=$BATCHED ##############################"
  env "${COMMON_ENV[@]}" GLM4_MOE_LITE_BATCHED_PREFILL="$BATCHED" \
    python models/experimental/glm4_moe_lite/scripts/ab_prefill_pcm_pcc.py \
      --simulate-context-len 128 --min-cache-tokens 256 \
      --mesh-rows 2 --mesh-cols 4 --kv-cache-dtype "$KVDT" \
      --batch-size "$B" --min-pcc 0.999 --time-iters 3 2>&1 | \
      grep -E "A/B TIME|A/B PCC|chunk_decision|RESULT|Traceback|Error:" | \
      grep -vE "total_tokens=32 " | sort -u
  echo "--- batch $B done (exit ${PIPESTATUS[0]}) ---"
}

run_batch 8  bf16 1
run_batch 16 bf16 1
run_batch 32 bf8  0

echo "=== DONE_MULTIBATCH ==="
