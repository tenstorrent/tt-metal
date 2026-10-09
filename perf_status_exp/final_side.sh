#!/bin/bash
# final_side.sh <tag> <mode>: full ISL sweep, both ops, glm_53/kimi_k2_7 x ndshard/interleaved.
W=/localdev/mbezulj/tt-metal/.claude/worktrees/agent-abb2aba9472191522
TAG=$1; MODE=$2
ISLS=0,32,64,96,128,160,192,224,256,288,320,384,448,512,544,640,768,800,1024
for M in glm_53 kimi_k2_7; do
  for L in w_ndshard w_interleaved; do
    TMO=1200 $W/perf_status_exp/run.sh ${TAG}_routed_${M}_${L} $MODE perf_status_exp/test_exp.py \
      "single_routed_expert and $M and $L" $ISLS
    TMO=1200 $W/perf_status_exp/run.sh ${TAG}_swiglu_${M}_${L} $MODE perf_status_exp/test_exp.py \
      "moe_fused_swiglu and $M and $L" $ISLS
  done
done
echo "SIDE_DONE $TAG $MODE" >> /localdev/mbezulj/logs/merge8_runs.txt
