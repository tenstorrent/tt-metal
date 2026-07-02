#!/usr/bin/env bash
# Iter-2 verification: A/B logits PCC (OLD 4-chunk vs NEW adaptive) + chunk-decision debug.
set -uo pipefail
cd /home/gtobar/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

TT_METAL_GTEST_ETH_DISPATCH=1 \
GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0 \
GLM4_MOE_LITE_CCL_NUM_LINKS=1 \
GLM4_MOE_LITE_CCL_TOPOLOGY=linear \
GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 GLM4_MOE_LITE_FUSE_QKV_A=1 \
GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 GLM4_MOE_LITE_BATCHED_PREFILL=1 \
GLM4_MOE_LITE_DECODE_L1_ACT=1 GLM4_MOE_LITE_EP_L1=1 \
GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1 GLM4_MOE_LITE_SKIP_TYPECAST=1 \
GLM4_MOE_LITE_MOE_CHUNK_DEBUG=1 \
python models/experimental/glm4_moe_lite/scripts/ab_prefill_pcm_pcc.py \
  --simulate-context-len 128 --min-cache-tokens 256 \
  --mesh-rows 2 --mesh-cols 4 --kv-cache-dtype bf16 --min-pcc 0.999 2>&1

echo "=== VERIFY_EXIT: $? ==="
echo "=== DONE_VERIFY ==="
