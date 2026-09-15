#!/usr/bin/env bash
# Source from the repository root before a fresh process. Investigation only.
source experiments/sdpa-l2/fp32-pipeline-v2/candidate-env.sh
export TT_SDPA_FP32_REFINE_MACRO=1
unset TT_SDPA_FP32_L1_PIPELINE TT_SDPA_FP32_LIGHT_RELOAD
