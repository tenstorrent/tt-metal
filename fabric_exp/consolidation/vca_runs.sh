#!/bin/bash
# VC1 allocation sweep, VC0 side (2026-10-28): v4gi -O3 on the VC1 split (vc1/vc1_alloc.yaml: multi-mesh sanity,
# data checked, then intra-mesh benchmark rows: 1-hop E/W and N/S, multicast, 2-hop through a VC0 forwarding channel,
# 2-hop merged with 1-hop flows, bursty 1-hop), hang capture armed, one run per allocation (fresh cache each), the
# whole list twice, the second time in reverse order (drift shows as a p1/p2 difference).
# Allocations, TT_FABRIC_EXP_SLOTS_VC1=worker,forwarding0,receiver0,forwarding1,receiver1 (0 = the rest to the worker):
#   main    the tier table: worker 51, VC0 forwarding 4, VC0 receiver 8, VC1 forwarding 2, VC1 receiver 4
#   r16     0,4,16,2,4    the VC0 receiver only (worker 43)
#   f8      0,8,8,2,4     VC0 forwarding only (worker 39)
#   f8r16   0,8,16,2,4    both (worker 31)
#   f8r16v  0,8,16,4,8    both, VC1 doubled (worker 21; correctness of VC1 at T = 2, its rows are not measurable)
#   f12r16  0,12,16,2,4   wider forwarding (worker 19)
#   f8r32   0,8,32,2,4    wider receiver (worker 15)
# Labels vca_<allocation>_p<pass>_o3.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
export VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 TT_FABRIC_EXP_LOG_SLOTS=1
declare -A SPLIT=([main]="" [r16]=0,4,16,2,4 [f8]=0,8,8,2,4 [f8r16]=0,8,16,2,4 [f8r16v]=0,8,16,4,8
                  [f12r16]=0,12,16,2,4 [f8r32]=0,8,32,2,4)
ORDER="main r16 f8 f8r16 f8r16v f12r16 f8r32"
python3 $CON/swap.py apply fabric_exp/variants/v4gi $BK || { echo STOPPED; exit 1; }
for p in 1 2; do
    list=$ORDER
    [ $p = 2 ] && list=$(echo $ORDER | tr ' ' '\n' | tac | tr '\n' ' ')
    for a in $list; do
        if [ -n "${SPLIT[$a]}" ]; then export TT_FABRIC_EXP_SLOTS_VC1=${SPLIT[$a]}; else unset TT_FABRIC_EXP_SLOTS_VC1; fi
        echo "##### $a pass $p: TT_FABRIC_EXP_SLOTS_VC1=${TT_FABRIC_EXP_SLOTS_VC1:-unset} (job at ${SECONDS} s)"
        VC1_RUN=run_vc1_cap vc1_variant vca_$a tree vca_${a}_p$p O3
        sed 's/\x1b\[[0-9;]*m//g' $out/vca_${a}_p${p}_o3_13129088_vc1.log | grep -a 'sender slots' \
            | grep -v '{4, 0, 0, 0}\|{0, 0, 0}' | sed 's/.*\(VC[0-9] sender slots [^(]*\).*/   \1/' | sort -u
    done
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
