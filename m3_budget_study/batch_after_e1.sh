#!/bin/bash
cd "$(dirname "$0")"
until ! pgrep -f "batch_e1[.]sh" >/dev/null; do sleep 20; done
declare -A L=([S8]=8,9,10,11,12,13,14,15)
for W in 4096 6144; do RUN_ID=e1_s8_w${W}_r2 EXP=E1 LAYER_SET=S8 BUDGET_LAYER_IDS=${L[S8]} BUDGET_W=$W BUDGET_POINTS=0:$W NOTES="rerun: first attempt broken by a concurrent session" ./run_budget.sh; done
./batch_e2.sh
./batch_e3.sh
