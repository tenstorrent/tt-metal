#!/bin/bash
# s6x.sh <tag>: step-6 recheck, swiglu kimi_k2_7/glm_53 ndshard, m8 vs new alternating, 2 rounds.
W=/localdev/mbezulj/tt-metal/.claude/worktrees/agent-abb2aba9472191522/perf_status_exp
T=$1
for i in 1 2; do
  for MODE in m8 new; do
    for M in kimi_k2_7 glm_53; do
      $W/run.sh ${T}${i}_swiglu_${M}_w_ndshard $MODE perf_status_exp/test_exp.py \
        "moe_fused_swiglu and $M and w_ndshard" 0,128,256,320
    done
  done
done
echo "S6X_DONE $T" >> /localdev/mbezulj/logs/merge8_runs.txt
