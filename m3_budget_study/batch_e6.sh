#!/bin/bash
# E6: SP=2 (2,4) with 15 sparse layers vs SP=4 (4,4) with S8, W in {4096, 8192} x h in {0, 139264, 548864}.
# 139264 stands in for 141312 (h must be a multiple of W on the plain path).
cd "$(dirname "$0")"
SP2=15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
S8=8,9,10,11,12,13,14,15
for W in 4096 8192; do
  P=0:$W,139264:$W,548864:$W
  RUN_ID=e6_sp2_w$W EXP=E6 LAYER_SET=SP2_15 BUDGET_STAGES=4 BUDGET_STAGE=1 BUDGET_LAYER_IDS=$SP2 BUDGET_W=$W BUDGET_POINTS=$P ./run_budget.sh
  RUN_ID=e6_sp4_w$W EXP=E6 LAYER_SET=S8 BUDGET_STAGES=2 BUDGET_STAGE=0 BUDGET_LAYER_IDS=$S8 BUDGET_W=$W BUDGET_POINTS=$P ./run_budget.sh
done
