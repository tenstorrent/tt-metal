#!/bin/bash
# Send-batch study (V4_SEND_BATCH.md, 2026-11-01): row A/B in one job. ROUTERS='a b ...' [REPS=1] [EXTRA_ENV=...]
# sb_rows_runs.sh TAG ROW...: each router (variant directory) swapped in once, fresh kernel cache per router and tag,
# then every ROW run REPS times. ROW: a fabric_exp/pcs3 row name, bnd/<name> (consolidation/bnd), sb/<name>
# (consolidation/sb), or lat (fabric_exp/exp_latency.yaml, latency mode). Labels sb<TAG>_<router>_<row>_<rep> (row with
# '/' -> '-'): summary CSV, per-device/direction CSV (_detail.csv), latency CSV (_latency.csv), log (per-sender lines).
# Report: sb_rows_report.py TAG. Hang capture armed; a hang or a failed test stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS \
    TT_FABRIC_EXP_SLOTS_VC2
T=$1; shift
[ -n "$T" ] && [ $# -gt 0 ] && [ -n "$ROUTERS" ] || { echo "usage: ROUTERS='a b' sb_rows_runs.sh TAG ROW..."; exit 1; }
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
yaml_of() {
    case $1 in
        lat) echo fabric_exp/exp_latency.yaml ;;
        bnd/*|sb/*) echo $CON/$1.yaml ;;
        *) echo fabric_exp/pcs3/$1.yaml ;;
    esac
}
for row in "$@"; do [ -f "$(yaml_of $row)" ] || { echo "no yaml for row $row"; exit 1; }; done
for r in $ROUTERS; do
    [ -d fabric_exp/variants/$r ] || { echo "no variant $r"; exit 1; }
done
for r in $ROUTERS; do
    c=/home/aagarwal/.tt-metal-cache-sb-$T-$r
    rm -rf "${c:?}"; mkdir -p $c
    python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
    for rep in $(seq 1 ${REPS:-1}); do
        for row in "$@"; do
            lab=sb${T}_${r}_${row//\//-}_$rep
            rm -f generated/fabric/bandwidth_results_blackhole.csv
            run_cap $lab $(yaml_of $row) $c 300 $EXTRA_ROW_ENV
            rc=$?
            [ -f generated/fabric/bandwidth_results_blackhole.csv ] && \
                cp generated/fabric/bandwidth_results_blackhole.csv $out/${lab}_detail.csv
            [ $rc = 0 ] || stop  # a hang (2) or a failed test (1) stops the job
        done
    done
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
