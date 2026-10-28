#!/bin/bash
# v4 after the send result as plain values and out-of-line credits: validation (mesh, val1d without rings/mux/telemetry), the family-1 1D suite at
# worker depth 16 and 24, and the 2D 4352 suite. Fresh cache.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
C=/home/aagarwal/.tt-metal-cache-v4g
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4 $BK || { echo STOPPED; exit 1; }
run v4g_13129088_mesh $(suite_yaml mesh) $C O3 || { echo STOPPED; exit 1; }
run v4g_13129088_val1d $(suite_yaml val1d_m1) $C O3 || { echo STOPPED; exit 1; }
run v4gw16_13129088_1d $(suite_yaml 1d_m1) $C O3
EXTRA_ENV="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24" run v4gw24_13129088_1d $(suite_yaml 1d_m1) $C O3
run v4g_13129088_4352 $(suite_yaml 4352) $C O3
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
for f in fabric_exp/results/v4gw16_13129088_1d.csv fabric_exp/results/v4gw24_13129088_1d.csv fabric_exp/results/v4g_13129088_4352.csv; do
    [ -f $f ] && awk -F, -v f=$(basename $f .csv) 'NR>1 {printf "%-24s %-34s %.1f\n", f, $1, 1.35e9/$10}' $f
done
rm -f $M
EXTRA_ENV="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24" bash $CON/pcsv_runs.sh $V/v4 v4g1d 1D lin1 lin3 linmcast
bash $CON/pcsv_runs.sh $V/v4 v4g2d 2D ew4352 hop3
