#!/bin/bash
# VC1 allocation, round 3 (2026-10-28): the VC0 / VC1 forwarding split at worker 19, VC0 receiver 16, VC1 receiver 4,
# v4gi -O3, plus a mixed VC0 + VC1 load. Per allocation, one fresh cache each:
#   1. vc1/vc1_alloc.yaml (multi-mesh sanity, data checked, then the intra-mesh rows), hang capture armed;
#      labels vcc_v4gi_<allocation>_p<pass>_o3
#   2. vc1/vc1_inter.yaml ([0, 0] to each chip of mesh 1); labels vcci_v4gi_<allocation>_p<pass>_o3
#   3. vc1/vc1_mix.yaml (a VC1 flow sharing [1, 1]'s W link with a VC0 worker or forwarded flow, and those VC0 flows
#      alone); labels vccm_v4gi_<allocation>_p<pass>_o3
# 2 and 3 without capture (run_vc1, 180 s timeout); after a failure the job skips that yaml.
# Allocations (TT_FABRIC_EXP_SLOTS_VC1, 0 = the rest to the worker, which is 19 for every a* below):
#   main  51,4,8,2,4 (the tier table)
#   a12   0,12,16,2,4     a10  0,10,16,4,4     a9  0,9,16,5,4     a8  0,8,16,6,4     a7  0,7,16,7,4
# Pass 1 in that order, then a10 and main again (pass 2) for drift.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
export TT_FABRIC_EXP_LOG_SLOTS=1
declare -A SPLIT=([main]="" [a12]=0,12,16,2,4 [a10]=0,10,16,4,4 [a9]=0,9,16,5,4 [a8]=0,8,16,6,4 [a7]=0,7,16,7,4)
INTER=1
MIX=1
slots() {  # log
    sed 's/\x1b\[[0-9;]*m//g' $1 | grep -a 'sender slots' | grep -v '{4, 0, 0, 0}\|{0, 0, 0}' \
        | sed 's/.*\(VC[0-9] sender slots [^(]*\).*/   \1/' | sort -u
}
one() {  # allocation pass
    local a=$1 p=$2 c=/home/aagarwal/.tt-metal-cache-p1vc1-vcc
    if [ -n "${SPLIT[$a]}" ]; then export TT_FABRIC_EXP_SLOTS_VC1=${SPLIT[$a]}; else unset TT_FABRIC_EXP_SLOTS_VC1; fi
    echo "##### $a pass $p: TT_FABRIC_EXP_SLOTS_VC1=${TT_FABRIC_EXP_SLOTS_VC1:-unset} (job at ${SECONDS} s)"
    VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 run_vc1_cap vcc_v4gi_${a}_p${p}_o3 $c O3 \
        || { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
    slots $out/vcc_v4gi_${a}_p${p}_o3_13129088_vc1.log
    if [ $INTER = 1 ]; then
        VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 run_vc1 vcci_v4gi_${a}_p${p}_o3 $c O3 \
            || { INTER=0; echo "!!! inter-mesh rows failed: skipped from here"; }
    fi
    if [ $MIX = 1 ]; then
        VC1_YAML=fabric_exp/vc1/vc1_mix.yaml VC1_TIMEOUT=180 run_vc1 vccm_v4gi_${a}_p${p}_o3 $c O3 \
            || { MIX=0; echo "!!! mixed-load rows failed: skipped from here"; }
        echo "   mix rows: $(grep -c . $out/vccm_v4gi_${a}_p${p}_o3_13129088_vc1_tests.csv 2>/dev/null) CSV lines"
    fi
}
python3 $CON/swap.py apply fabric_exp/variants/v4gi $BK || { echo STOPPED; exit 1; }
for a in main a12 a10 a9 a8 a7; do one $a 1; done
for a in a10 main; do one $a 2; done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
