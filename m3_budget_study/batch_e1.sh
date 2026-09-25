#!/bin/bash
# E1 cold width sweep (plain, B=1) + §4.4 sanity widths.
cd "$(dirname "$0")"
declare -A L=([S8]=8,9,10,11,12,13,14,15 [D]=0,1,2 [S0]=0,1,2,3,4,5,6,7)
run () { RUN_ID=$1 EXP=$2 LAYER_SET=$3 BUDGET_LAYER_IDS=${L[$3]} BUDGET_W=$4 BUDGET_POINTS=$5 ./run_budget.sh; }
run s44_s8_w5120 SANITY S8 5120 0:5120
run s44_s8_w8192 SANITY S8 8192 0:8192
for W in 4096 6144 10240; do run e1_s8_w$W E1 S8 $W 0:$W; done
for LS in D S0; do
  run e1_$(echo $LS | tr A-Z a-z)_w2048 E1 $LS 2048 0:2048,0:1024,0:256
  for W in 4096 5120 6144 8192 10240; do run e1_$(echo $LS | tr A-Z a-z)_w$W E1 $LS $W 0:$W; done
done
