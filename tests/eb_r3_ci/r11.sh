#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, rules of 16:40 / 17:40: each kept caller edit against main's program at shapes Blackhole models run
# (test_eb_r11.py: Llama 3.1-8B P150 decoder layer and sampling, SDXL BH modules, r10 op replicas). ab_r10.sh: main optin optin
# main in one tree (path-matched), EB_REPS repetitions per test, per op configuration. usage: r11.sh <job>
cd /work
export EB_RUN_LIMIT=2400 EB_REPS=${EB_REPS:-4} TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
T=tests/eb_r3_ci/test_eb_r11.py; O=tests/eb_r3_ci/r11
ab() { echo "##### $(date -u +%T) $1: off=$2 main_env=$3 k=$4"; EB_K_EXPR="$4" bash tests/eb_r3_ci/ab_r10.sh "$2" "$3" -p eb_k_plugin ${5:-$T}; }
DS=tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill
case $1 in
  f6llama) ab f6 - EB_R3_PER_FACE=1 "llama8b_decode or llama8b_prefill" ;;
  f6ops)   ab f6 - EB_R3_PER_FACE=1 "test_mul_cfg or sdxl_transformer" ;;
  f1)      ab f1 - EB_R3_NO_BLOCK=1 "llama8b_decode or sdxl_resnet or sdxl_transformer" ;;
  f4f3)    ab f4 - EB_R3_NO_PRE_SECTIONS=1 "llama8b_decode"
           ab f3 - EB_R3_MAIN_REINIT=1 "sdxl_temb_add" ;;
  lnh)     ab ln_h $O/off_ln_h.txt - "llama8b_decode or sdxl_transformer" ;;
  lnb)     ab ln_b $O/off_ln_b.txt - "llama8b_decode or sdxl_transformer" ;;
  sdpa)    ab fd_h $O/off_fd_h.txt - "llama8b_decode"
           ab sdpa_b $O/off_sdpa_b.txt - "llama8b_prefill or sdxl_transformer" ;;
  lnint)   ab lnint_h $O/off_lnint_h.txt - "sdxl_transformer or llama8b_prefill"
           ab lnint_b $O/off_lnint_b.txt - "sdxl_transformer or llama8b_prefill" ;;
  dsp)     ab csa_dr $O/off_csa_dr.txt - "test_csa_compressor_single_device" models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_csa_compressor.py
           ab mhg_b $O/off_mhg_b.txt - "test_moe_hash_gate and realistic and pad0 and dsv4" $DS/test_moe_hash_gate.py
           ab mhc_h $O/off_mhc_h.txt - "test_mhc_split_sinkhorn and not sharded" $DS/test_mhc_split_sinkhorn.py ;;
  x)       ab f6x - EB_R3_PER_FACE=1 "sdxl_refiner_geglu or test_mul_tg or mul_bfp8"
           ab f1x - EB_R3_NO_BLOCK=1 "sdxl_refiner_geglu"
           ab f4x - EB_R3_NO_PRE_SECTIONS=1 "test_mul_tg" ;;
  x2)      for i in 1 2 3; do
             ab f6x2 - EB_R3_PER_FACE=1 "test_mul_tg or mul_bfp8 or (sdxl_refiner_geglu and d1)"
             ab f4x2 - EB_R3_NO_PRE_SECTIONS=1 "test_mul_tg or mul_bfp8"
           done ;;
  f1b)     for i in 1 2 3; do ab f1b - EB_R3_NO_BLOCK=1 "llama8b_decode"; done ;;
  lnint2)  for i in 1 2 3; do
             ab lnint_h $O/off_lnint_h.txt - "(sdxl_transformer and d1) or (llama8b_prefill and accuracy)"
             ab lnint_b $O/off_lnint_b.txt - "(sdxl_transformer and d1) or (llama8b_prefill and accuracy)"
           done ;;
  smsa)    ab sm_b $O/off_sm_b.txt - "softmax_cfg"
           ab sa_b $O/off_sa_b.txt - "llama8b_sampling" ;;
esac
echo "##### end $(date -u +%T)"
