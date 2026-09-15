#!/usr/bin/env bash
# Source before a fresh Python process. Narrow investigation guard forbids
# fallback: Blackhole BF16 Q/K/V, public HiFi2+approx request overridden to
# HiFi4, FP32 DST, noncausal D128, Q128 K512/1024, unpadded N>=32768.
export TT_SDPA_ACCURACY_DIAG=4
export TT_SDPA_FP32_SUB_BATCH=2
export TT_SDPA_FP32_FUSED_EXP=1
export TT_SDPA_FP32_REUSE_EXP=1
export TT_SDPA_FP32_EXTRA_CONST=1
export TT_SDPA_FP32_PAIRED_UNPACK=1
export TT_SDPA_FP32_PAIRED_PACK=1
# Exact zero/one SrcA: phases 0+2, not ordinary HiFi2. QK/PV remain HiFi4.
export TT_SDPA_DENOM_PHASES=2
unset TT_SDPA_FP32_CACHE_MAX TT_SDPA_FP32_SHADOW_MAX
unset TT_SDPA_FP32_QK_WIDTH TT_SDPA_FP32_QK_HEIGHT TT_SDPA_UTIL_PROFILE
