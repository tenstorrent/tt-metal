#!/bin/bash
# Row-first A/B (2026-10-30, user: before a full census, A/B the rows that lost the most). row_ab_runs.sh TAG ROW...:
# each router in ROUTERS (variant directories) swapped in once, then every ROW (fabric_exp/pcs3/<row>.yaml, one test
# each) run REPS times (default 2), fresh kernel cache per router. Labels rab<TAG>_<router>_<row>_<rep>; compare with
# row_ab_report.py TAG. A hang stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
T=$1; shift
[ -n "$T" ] && [ $# -gt 0 ] && [ -n "$ROUTERS" ] || { echo "usage: ROUTERS='a b' row_ab_runs.sh TAG ROW..."; exit 1; }
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
for r in $ROUTERS; do
    c=/home/aagarwal/.tt-metal-cache-rab-$T-$r
    rm -rf "${c:?}"; mkdir -p $c
    python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
    for rep in $(seq 1 ${REPS:-2}); do
        for row in "$@"; do
            run_cap rab${T}_${r}_${row}_$rep fabric_exp/pcs3/$row.yaml $c 200 $EXTRA_ROW_ENV
            [ $? = 2 ] && stop
        done
    done
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
