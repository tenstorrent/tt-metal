#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, final head merged with main: layernorm.cpp's two defines removed against kept at SDXL's L1 layer
# norms and Llama 3.1-8B's prefill RMSNorm, three passes.
cd /work
export EB_RUN_LIMIT=2400 EB_REPS=4 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
for i in 1 2 3; do
  echo "##### $(date -u +%T) lnint_hb: off=tests/eb_r3_ci/r12/off_lnint.txt main_env=- k=sdxl_transformer and d1"
  EB_K_EXPR="(sdxl_transformer and d1) or (llama8b_prefill and performance and 2048)" bash tests/eb_r3_ci/ab_r10.sh tests/eb_r3_ci/r12/off_lnint.txt - -p eb_k_plugin tests/eb_r3_ci/test_eb_r11.py
done
echo "##### end $(date -u +%T)"
