#!/bin/bash
# ab.sh <tag> <mode...>: step A/B matrix (routed 0,256,512,1024; swiglu 0,64,128,256,320), 4 series per mode.
W=/localdev/mbezulj/tt-metal/.claude/worktrees/agent-abb2aba9472191522
TAG=$1; shift
for MODE in "$@"; do
  for M in glm_53 kimi_k2_7; do
    for L in w_ndshard w_interleaved; do
      $W/perf_status_exp/run.sh ${TAG}_routed_${M}_${L} $MODE perf_status_exp/test_exp.py \
        "single_routed_expert and $M and $L" 0,256,512,1024
      $W/perf_status_exp/run.sh ${TAG}_swiglu_${M}_${L} $MODE perf_status_exp/test_exp.py \
        "moe_fused_swiglu and $M and $L" 0,64,128,256,320
    done
  done
  echo "AB_DONE $TAG $MODE" >> /localdev/mbezulj/logs/merge8_runs.txt
done
