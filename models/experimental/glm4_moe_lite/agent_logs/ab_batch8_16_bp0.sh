source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
COMMON="TT_METAL_GTEST_ETH_DISPATCH=1 GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0 GLM4_MOE_LITE_CCL_NUM_LINKS=1 GLM4_MOE_LITE_CCL_TOPOLOGY=linear GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 GLM4_MOE_LITE_FUSE_QKV_A=1 GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 GLM4_MOE_LITE_DECODE_L1_ACT=1 GLM4_MOE_LITE_EP_L1=1 GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1 GLM4_MOE_LITE_SKIP_TYPECAST=1 GLM4_MOE_LITE_MOE_CHUNK_DEBUG=1 GLM4_MOE_LITE_BATCHED_PREFILL=0"
for B in 8 16; do
  echo "########## BATCH=$B BATCHED_PREFILL=0 ##########"
  env $COMMON python models/experimental/glm4_moe_lite/scripts/ab_prefill_pcm_pcc.py \
    --simulate-context-len 128 --min-cache-tokens 256 --mesh-rows 2 --mesh-cols 4 \
    --kv-cache-dtype bf16 --batch-size $B --min-pcc 0.999 --time-iters 3 2>&1 | \
    grep -E "A/B TIME|A/B PCC|RESULT|chunk_decision|Traceback|FATAL|THROW|RuntimeError" | grep -vE "total_tokens=32 " | sort -u
  echo "--- batch $B exit ${PIPESTATUS[0]} ---"
done
echo "=== DONE_BP0 ==="
