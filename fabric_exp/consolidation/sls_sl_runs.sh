#!/bin/bash
# N/S floor, mechanism A/B on the P150 (SLOT_STUDY.md, 2026-11-02): variants/v4bdd (the lead + delay hooks, = the lead at
# 0) against variants/v4bddsl (DIAGNOSTIC: every VC0 forward stateless, as the Galaxy's N/S routers forward), on the 2D
# and 1D rows (fabric_exp/slotstudy/sls_{2d,1d}_p150.yaml, builder default slots: 2D 17/16/16, 1D 26/25/32), arms base /
# tx32 / rx32 (a 32-cycle spin per packet sent / received). Labels sls<PASS>sl_<router>_<arm>_<2d|1d> (.log, .csv,
# _tests.csv). Slots logged. A failure or hang stops the job.
#   sls_sl_runs.sh PASS ROUTER...
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
P=$1; shift
[ -n "$P" ] && [ $# -gt 0 ] || { echo "usage: sls_sl_runs.sh PASS ROUTER..."; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
arm_env() { case $1 in base) ;; tx*) echo "TT_FABRIC_EXP_TX_DELAY=${1#tx}" ;; rx*) echo "TT_FABRIC_EXP_RX_DELAY=${1#rx}" ;; esac; }
for r in "$@"; do
    python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
    for a in ${ARMS:-base tx32 rx32}; do
        rm -rf /home/aagarwal/.tt-metal-cache-slssl-$r-$a
        for s in 2d 1d; do
            [ $SECONDS -gt ${SLS_BUDGET:-1100} ] && { echo "=== budget: stopping at ${SECONDS} s"; python3 $CON/swap.py restore $BK; exit 0; }
            l=sls${P}sl_${r}_${a}_$s; C=/home/aagarwal/.tt-metal-cache-slssl-$r-$a
            n=14; [ $s = 1d ] && n=6
            rm -f generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv
            run_cap $l fabric_exp/slotstudy/sls_${s}_p150.yaml $C 300 TT_FABRIC_EXP_LOG_SLOTS=1 $(arm_env $a); rc=$?
            [ -f generated/fabric/bandwidth_results_blackhole.csv ] && cp generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv
            grep -aq "Tests: $n executed, 0 failed" $out/$l.log || rc=1
            [ $rc = 0 ] || { echo "!!! $l failed (rc $rc)"; stop; }
        done
    done
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
