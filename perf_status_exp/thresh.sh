#!/bin/bash
# thresh.sh <tag> <mode>: swiglu vs routed, both on 11x8, ndshard, around the hybrid threshold.
W=/localdev/mbezulj/tt-metal/.claude/worktrees/agent-abb2aba9472191522
TAG=$1; MODE=${2:-new}
for M in glm_53 kimi_k2_7; do
  $W/perf_status_exp/run.sh ${TAG}_routed_${M}_w_ndshard $MODE perf_status_exp/test_exp.py \
    "single_routed_expert and $M and w_ndshard" 192,224,256,272,288,320,352,384
  $W/perf_status_exp/run.sh ${TAG}_swiglu_${M}_w_ndshard $MODE perf_status_exp/test_exp.py \
    "moe_fused_swiglu and $M and w_ndshard" 192,224,256,272,288,320,352,384
done
echo "THRESH_DONE $TAG" >> /localdev/mbezulj/logs/merge8_runs.txt
