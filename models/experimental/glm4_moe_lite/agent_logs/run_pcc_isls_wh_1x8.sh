#!/usr/bin/env bash
# TP=1 1x8 prefill-logits PCC vs HF across ISLs (real corpus). HF refs are cached under
# experiments/hf_ref, so this only runs TT prefill + compares (fast).
set -u
cd "$(git rev-parse --show-toplevel)"
source python_env/bin/activate

export GLM4_MOE_LITE_PCC_ISLS="${GLM4_MOE_LITE_PCC_ISLS:-128 512 1024 2048 4096 8192}"
export TT_METAL_GTEST_ETH_DISPATCH=1
export GLM4_MOE_LITE_CCL_NUM_LINKS=1
export GLM4_MOE_LITE_CCL_TOPOLOGY=ring
export GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0

python -m pytest -q -s -p no:randomly \
  models/experimental/glm4_moe_lite/tests/pipeline_tests/test_text_prefill_logits_wh.py \
  -k "tp1_1x8"
