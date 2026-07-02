#!/usr/bin/env bash
# Iter-2 reprofile: adaptive prefill_pcm (1-chunk ISL-128). Batch-1 ISL-128 prefill.
# Produces ops_perf_results CSV for tt-perf-report + generated tokens for correctness.
set -uo pipefail
cd /home/gtobar/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

OUT=/tmp/glm_tracy_iter2
rm -rf "$OUT"

TT_METAL_GTEST_ETH_DISPATCH=1 \
GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0 \
GLM4_MOE_LITE_CCL_NUM_LINKS=1 \
GLM4_MOE_LITE_CCL_TOPOLOGY=linear \
GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 GLM4_MOE_LITE_FUSE_QKV_A=1 \
GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 GLM4_MOE_LITE_BATCHED_PREFILL=1 \
GLM4_MOE_LITE_DECODE_L1_ACT=1 GLM4_MOE_LITE_EP_L1=1 \
GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1 GLM4_MOE_LITE_SKIP_TYPECAST=1 \
GLM4_MOE_LITE_SIGNPOST=1 \
python -m tracy -r -p -o "$OUT" -- \
  models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Summarize" --simulate-context-len 128 --min-cache-tokens 256 \
  --max-new-tokens 4 --batch-size 1 \
  --mesh-rows 2 --mesh-cols 4 --kv-cache-dtype bf16 \
  --phase prefill 2>&1

echo "=== RUN EXIT: $? ==="
echo "=== CSV ==="
find "$OUT" -name 'ops_perf_results_*.csv' 2>/dev/null
echo "=== DONE_ITER2 ==="
