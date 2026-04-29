#!/bin/bash
# Validate dynamic dispatch buffer on main: pie960 PCC + longdialogue smoke-50 at 25K.
# Both e256_cf32_device_fp32, balanced, right_pad.

MAIN=/data/ipotkonjak/single_glx_debug
ENV="TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/deepseek-prefill-cache/golden-32a3e8b6 TT_DS_PREFILL_TTNN_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill  DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528"
BASE="pretrained and e256_cf32_device_fp32 and mesh-8x4 and 61 and balanced and right_pad"

run() {
    local k="$1" log="$2"
    cd $MAIN && source python_env/bin/activate
    eval "$ENV pytest -vs models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py -k '$k' |& tee $log"
}

# pie_R_DDB: PCC, pie960, 1K
run "$BASE and pcc and iter1 and 1024 and pie960"                    $MAIN/250428_de36a93f_pie_bal_R

# smk_25k_fp32: smoke 50 iters, longdialogue, 25K
run "$BASE and smoke and iter50 and 25600 and longdialogue_qa_eng"   $MAIN/250428_ddb_smoke50_ldg_25k_fp32
