#!/bin/bash
# E2c capacity sweep (does cost follow KV capacity at fixed h?), then E2 single-segment depth grid, then S0 check.
cd "$(dirname "$0")"
declare -A L=([S8]=8,9,10,11,12,13,14,15 [D]=0,1,2 [S0]=0,1,2,3,4,5,6,7)
run () { RUN_ID=$1 EXP=$2 LAYER_SET=$3 BUDGET_LAYER_IDS=${L[$3]} BUDGET_W=$4 BUDGET_POINTS=$5 BUDGET_CAPACITY=${6:-0} ./run_budget.sh; }
for C in 18432 143360 550912 1048576; do run e2c_d_cap$C E2c D 2048 16384:2048,16384:256 $C; done
for C in 18432 1048576; do run e2c_s8_cap$C E2c S8 2048 16384:2048,16384:256 $C; done
PTS=""; for h in 0 16384 65536 141312 309248 548864; do for n in 256 1024 2048; do PTS+="$h:$n,"; done; done
run e2_d_grid E2 D 2048 "$PTS"
run e2_s8_grid E2 S8 2048 "$PTS"
run e2_s0_check E2 S0 2048 0:2048,141312:2048,548864:2048
