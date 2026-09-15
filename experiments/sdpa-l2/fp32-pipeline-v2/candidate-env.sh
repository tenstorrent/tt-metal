#!/usr/bin/env bash
# Source from the repository root before a fresh process. Investigation only:
# the factory rejects unsupported inputs rather than silently falling back.
source experiments/sdpa-l2/fp32-util-v1/candidate-env.sh
export TT_SDPA_FP32_L1_SUB=1
export TT_SDPA_FP32_L1_MACRO=1
unset TT_SDPA_FP32_PIPELINE TT_SDPA_FP32_L1_REPEAT
