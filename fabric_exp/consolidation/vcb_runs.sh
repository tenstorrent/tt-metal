#!/bin/bash
# VC1 allocation, round 2 (2026-10-28): v4gi against v4git (v4gi with each completion group sized to the smaller of
# the ring and the smallest far sender channel of its VC, instead of the ring alone), interleaved per allocation, -O3,
# one run each (vca's pass-to-pass spread was <= 0.7 cycles/packet), main's allocation first and last for drift.
# Per allocation and variant:
#   1. vc1/vc1_alloc.yaml (multi-mesh sanity, data checked, then the intra-mesh rows), hang capture armed;
#      labels vcb_<variant>_<allocation>_p1_o3
#   2. vc1/vc1_inter.yaml: [0, 0] to each chip of mesh 1 (the inter-mesh link, then VC1 inside mesh 1), the first runs
#      of the BandwidthProfiler fix for configs without hops; no capture (run_vc1, 180 s timeout); after a failure the
#      job skips these rows; labels vcbi_<variant>_<allocation>_p1_o3
# Allocations (TT_FABRIC_EXP_SLOTS_VC1=worker,forwarding0,receiver0,forwarding1,receiver1, 0 = the rest to the worker):
#   main      51, 4, 8, 2, 4   the tier table
#   f8        0,8,8,2,4        worker 39
#   f8r16     0,8,16,2,4       worker 31
#   f12r16    0,12,16,2,4      worker 19
#   f8r16v    0,8,16,4,8       worker 21 (VC1 doubled)
#   f10r16v4  0,10,16,4,4      worker 19 (VC1 forwarding 4, receiver 4)
#   f8r16v6   0,8,16,6,4       worker 19 (VC1 forwarding 6, receiver 4)
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
export TT_FABRIC_EXP_LOG_SLOTS=1
declare -A SPLIT=([main]="" [f8]=0,8,8,2,4 [f8r16]=0,8,16,2,4 [f12r16]=0,12,16,2,4 [f8r16v]=0,8,16,4,8
                  [f10r16v4]=0,10,16,4,4 [f8r16v6]=0,8,16,6,4)
ORDER="main f8 f8r16 f12r16 f8r16v f10r16v4 f8r16v6"
INTER=1
slots() {  # log
    sed 's/\x1b\[[0-9;]*m//g' $1 | grep -a 'sender slots' | grep -v '{4, 0, 0, 0}\|{0, 0, 0}' \
        | sed 's/.*\(VC[0-9] sender slots [^(]*\).*/   \1/' | sort -u
}
one() {  # variant allocation pass
    local v=$1 a=$2 p=$3 c=/home/aagarwal/.tt-metal-cache-p1vc1-vcb
    if [ -n "${SPLIT[$a]}" ]; then export TT_FABRIC_EXP_SLOTS_VC1=${SPLIT[$a]}; else unset TT_FABRIC_EXP_SLOTS_VC1; fi
    echo "##### $v $a: TT_FABRIC_EXP_SLOTS_VC1=${TT_FABRIC_EXP_SLOTS_VC1:-unset} (job at ${SECONDS} s)"
    python3 $CON/swap.py apply fabric_exp/variants/$v $BK || { echo STOPPED; exit 1; }
    VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 run_vc1_cap vcb_${v}_${a}_p${p}_o3 $c O3 \
        || { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
    slots $out/vcb_${v}_${a}_p${p}_o3_13129088_vc1.log
    if [ $INTER = 1 ]; then
        VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 run_vc1 vcbi_${v}_${a}_p${p}_o3 $c O3 \
            || { INTER=0; echo "!!! inter-mesh rows failed: skipped from here"; }
        echo "   inter rows: $(grep -c . $out/vcbi_${v}_${a}_p${p}_o3_13129088_vc1_tests.csv 2>/dev/null) CSV lines"
    fi
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
}
for a in $ORDER; do
    for v in v4gi v4git; do one $v $a 1; done
done
for v in v4gi v4git; do one $v main 2; done
rm -f $M
echo ALL_DONE
