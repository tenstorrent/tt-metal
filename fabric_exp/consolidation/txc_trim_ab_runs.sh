#!/bin/bash
# Sender consolidation C1b (V4_SENDER_CONSOLIDATION.md; 2026-11-03): same-job A/B of two routers on the channel-trimmed
# rows whose replays have forwarding-only routers (txc/trim_fwd_4352.yaml: Mesh3HopEW and MeshMulticast from
# exp_2d_trim_4352.yaml, each a capture pass then a trimmed replay). txc_trim_ab_runs.sh TAG A B: passes A, B, B, A
# (two passes in reversed order), fresh kernel cache per router. Labels txt<TAG>_<router>_<pass>; compare with
# txc_trim_report.py TAG. A hang stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
T=$1 A=$2 B=$3
[ -n "$T" ] && [ -d fabric_exp/variants/$A ] && [ -d fabric_exp/variants/$B ] || { echo "usage: txc_trim_ab_runs.sh TAG A B"; exit 1; }
Y=fabric_exp/consolidation/txc/trim_fwd_4352.yaml
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
for r in $A $B; do c=/home/aagarwal/.tt-metal-cache-txt-$T-$r; rm -rf "${c:?}"; mkdir -p $c; done
n=0
for r in $A $B $B $A; do
    n=$((n + 1))
    echo "##### pass $n: $r (job at ${SECONDS} s)"
    python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
    run_cap txt${T}_${r}_$n $Y /home/aagarwal/.tt-metal-cache-txt-$T-$r 300
    [ $? = 2 ] && stop
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
