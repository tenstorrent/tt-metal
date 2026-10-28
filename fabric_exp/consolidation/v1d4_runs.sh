#!/bin/bash
# 1D analysis, v4 (RESULTS §98): the family-1 1D suite at worker depth 16 and 24 (TT_FABRIC_EXP_LINEAR_SPARE_SLOTS +
# TT_FABRIC_EXP_WORKER_SLOTS), then PC sampling of the 1D cost-map rows at depth 24.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
C=/home/aagarwal/.tt-metal-cache-1d-v4
python3 $CON/swap.py apply $V/v4 $BK || { echo STOPPED; exit 1; }
run v4w16_13129088_1d $(suite_yaml 1d_m1) $C O3
EXTRA_ENV="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24" run v4w24_13129088_1d $(suite_yaml 1d_m1) $C O3
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
for f in fabric_exp/results/v4w16_13129088_1d.csv fabric_exp/results/v4w24_13129088_1d.csv; do
    [ -f $f ] && awk -F, -v f=$(basename $f .csv) 'NR>1 {printf "%-24s %-34s %.1f\n", f, $1, 1.35e9/$10}' $f
done
rm -f $M
EXTRA_ENV="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24" bash $CON/pcsv_runs.sh $V/v4 v4d1d 1D lin1 lin3
