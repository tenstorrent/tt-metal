#!/bin/bash
# E2w: separate compute (∝ W·h) from history-only (∝ h) terms: same depths at wider W.
cd "$(dirname "$0")"
until ! pgrep -f "batch_after_e1[.]sh" >/dev/null; do sleep 20; done
declare -A L=([S8]=8,9,10,11,12,13,14,15 [D]=0,1,2)
for LS in D S8; do for W in 4096 8192; do
  RUN_ID=e2w_$(echo $LS | tr A-Z a-z)_w$W EXP=E2w LAYER_SET=$LS BUDGET_LAYER_IDS=${L[$LS]} BUDGET_W=$W \
    BUDGET_POINTS=0:$W,65536:$W,548864:$W,548864:2048 ./run_budget.sh
done; done
