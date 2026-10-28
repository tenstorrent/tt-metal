#!/bin/bash
# v4 with the worker credit out of line, the upstream credit inline: validation (mesh, val1d without rings/mux/telemetry), the family-1 1D suite at
# worker depth 16 and 24, and the 2D 4352 suite. Fresh cache.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
C=/home/aagarwal/.tt-metal-cache-v4h
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4 $BK || { echo STOPPED; exit 1; }
run v4h_13129088_mesh $(suite_yaml mesh) $C O3 || { echo STOPPED; exit 1; }
run v4h_13129088_val1d $(suite_yaml val1d_m1) $C O3 || { echo STOPPED; exit 1; }
run v4hw16_13129088_1d $(suite_yaml 1d_m1) $C O3
EXTRA_ENV="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24" run v4hw24_13129088_1d $(suite_yaml 1d_m1) $C O3
run v4h_13129088_4352 $(suite_yaml 4352) $C O3
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
for f in fabric_exp/results/v4hw16_13129088_1d.csv fabric_exp/results/v4hw24_13129088_1d.csv fabric_exp/results/v4h_13129088_4352.csv; do
    [ -f $f ] && awk -F, -v f=$(basename $f .csv) 'NR>1 {printf "%-24s %-34s %.1f\n", f, $1, 1.35e9/$10}' $f
done
rm -f $M
EXTRA_ENV="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24" bash $CON/pcsv_runs.sh $V/v4 v4h1d 1D lin1 lin3 linmcast
bash $CON/pcsv_runs.sh $V/v4 v4h2d 2D ew4352 hop3
