#!/bin/bash
# VC2 traffic rows (2026-11-01): VC0 alone (ew4352), VC2 alone (vc2ew4352), VC0 + VC2 on one link (vc02ew4352: VC0
# finishes first; vc02bew4352: VC2 finishes first; per-sender rates from the profiler's log lines), all
# with VC2 on, per router, at two slot splits:
#   d   the study row (VC0 17/14/16, VC2 2/4)
#   v8  a deeper VC2 (TT_FABRIC_EXP_SLOTS_VC2=17,10,16,8,10: VC0 17/10/16, VC2 8/10; 1-hop rows use no forwarding)
# ROUTERS='a b' vc2rows_runs.sh TAG; REPS (default 2). Labels v2t<TAG>_<router>_<split>_<row>_<rep>: the summary CSV
# (.csv, the link's combined rate for vc02) and the per-device/direction CSV (_detail.csv). Slots logged. A hang stops
# the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS \
    TT_FABRIC_EXP_SLOTS_VC2
T=$1
[ -n "$T" ] && [ -n "$ROUTERS" ] || { echo "usage: ROUTERS='a b' vc2rows_runs.sh TAG"; exit 1; }
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
VC2="TT_METAL_ENABLE_FABRIC_VC2=1 TT_FABRIC_EXP_LOG_SLOTS=1"
for r in $ROUTERS; do
    c=/home/aagarwal/.tt-metal-cache-v2t-$T-$r
    rm -rf "${c:?}"; mkdir -p $c
    python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
    for rep in $(seq 1 ${REPS:-2}); do
        for split in d v8; do
            senv=""
            [ $split = v8 ] && senv="TT_FABRIC_EXP_SLOTS_VC2=17,10,16,8,10"
            for row in ${ROWS:-ew4352 vc2ew4352 vc02ew4352 vc02bew4352}; do
                lab=v2t${T}_${r}_${split}_${row}_$rep
                rm -f generated/fabric/bandwidth_results_blackhole.csv
                run_cap $lab fabric_exp/pcs3/$row.yaml $c 300 $VC2 $senv
                rc=$?
                [ -f generated/fabric/bandwidth_results_blackhole.csv ] && \
                    cp generated/fabric/bandwidth_results_blackhole.csv $out/${lab}_detail.csv
                [ $rc = 2 ] && stop
            done
        done
    done
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
