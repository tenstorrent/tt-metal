#!/usr/bin/env bash
# Ground-truth PCC: TT prefill vs cached HF reference. First run computes+caches HF
# (~59GB CPU forward, minutes); later runs load the cache and are fast.
# Runs TT under the production flags (same config as the prefill_pcm optimization).
set -uo pipefail
cd /home/gtobar/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

SEQ_LEN="${1:-128}"

TT_METAL_GTEST_ETH_DISPATCH=1 \
GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0 \
GLM4_MOE_LITE_CCL_NUM_LINKS=1 \
GLM4_MOE_LITE_CCL_TOPOLOGY=linear \
GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 GLM4_MOE_LITE_FUSE_QKV_A=1 \
GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 GLM4_MOE_LITE_BATCHED_PREFILL=1 \
GLM4_MOE_LITE_DECODE_L1_ACT=1 GLM4_MOE_LITE_EP_L1=1 \
GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1 GLM4_MOE_LITE_SKIP_TYPECAST=1 \
python models/experimental/glm4_moe_lite/scripts/pcc_vs_hf.py \
  --seq-len "$SEQ_LEN" --mesh-rows 2 --mesh-cols 4 --kv-cache-dtype bf16 --min-pcc 0.97 2>&1

echo "=== PCC_VS_HF_EXIT: $? ==="
echo "=== DONE_PCC_HF ==="
