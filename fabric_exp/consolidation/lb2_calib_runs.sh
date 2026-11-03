#!/bin/bash
# Second-LB host calibration (LB2_HARNESS.md), run on the extra LB through lb2_run.sh: the rows a09u32 measured on
# 2026-11-02/03, with the same scripts and settings.
#   lb2_calib_runs.sh TAG
#   1. pcs3 rows ew4352 hop3 lin3, two reps each: main_fw with report_size_runs.sh's main_fw settings (kernel v1, flags 0,
#      firmware keepalive), labels lbc<TAG>_main_fw_<row>_<rep>; then v4rxg3 through row_ab_runs.sh (as jobs 493/494,
#      a09u32: 127.0 / 146.7 / 124.4 cycles/packet), labels rablbc<TAG>_v4rxg3_<row>_<rep>
#   2. report_size_runs.sh lbc<TAG> 2d 1d with ROUTERS "main_fw v4rxg3" (the a09u32 size sweep, jobs 465-468), labels
#      rszlbc<TAG>_<router>_<suite>
# A hang stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
T=$1
[ -n "$T" ] || { echo "usage: lb2_calib_runs.sh TAG"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
V1="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
c=/home/aagarwal/.tt-metal-cache-lbc$T-main_fw
rm -rf "${c:?}"; mkdir -p $c
for rep in 1 2; do
    for row in ew4352 hop3 lin3; do
        KA_ENV= run_cap lbc${T}_main_fw_${row}_$rep fabric_exp/pcs3/$row.yaml $c 200 $V1
        [ $? = 2 ] && { echo STOPPED; exit 1; }
    done
done
ROUTERS=v4rxg3 REPS=2 bash $CON/row_ab_runs.sh lbc$T ew4352 hop3 lin3 || { echo STOPPED; exit 1; }
ROUTERS="main_fw v4rxg3" bash $CON/report_size_runs.sh lbc$T 2d 1d || { echo STOPPED; exit 1; }
for f in fabric_exp/results/lbc${T}_main_fw_*.csv fabric_exp/results/rablbc${T}_v4rxg3_*.csv; do
    awk -F, -v f=$(basename $f .csv) 'NR > 1 {printf "%s: %s %.1f cycles/packet\n", f, $1, 1.35e9 / $10}' $f
done
for r in main_fw v4rxg3; do for s in 2d 1d; do
    awk -F, -v t="rszlbc${T}_${r}_$s" 'NR > 1 && $7 == 4352 && $3 == "NOC_UNICAST_WRITE" && $1 ~ /^(Mesh1HopEW|Mesh3HopEW|Linear1HopEW|Linear3HopEW)$/ {
        printf "%s: %s %.1f cycles/packet\n", t, $1, 1.35e9 / $10}' fabric_exp/results/rszlbc${T}_${r}_$s.csv
done; done
echo "ALL_DONE (job at ${SECONDS} s)"
