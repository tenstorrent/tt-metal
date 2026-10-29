#!/bin/bash
# Tensix mux mode throughput, main's router against v4tm (2026-10-28; job 126: the ring deadlock-check row 419.6 vs 440.9
# cycles/packet). exp_mux.yaml (1D, ring and 2D 1-hop / bidirectional / 3-hop / multicast rows and the full-ring
# multicast, 1024 and 4352 B, mux mode) twice per router in the order main, v4tm, main, v4tm, hang capture armed, one
# fresh cache per router (compiles are in setup, not in the measured window). Main: kernel v1, flags 0, firmware
# keepalive. Labels mx<pass>_{v1,v4tm}_13129088_mux (pass a, b); compare with
# same_job_compare.py mxa_v1 mxa_v4tm mxb_v1 mxb_v4tm --rows 5.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
L="TT_FABRIC_EXP_LOG_SLOTS=1"
MAIN="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
Y=fabric_exp/v4/exp_mux.yaml
C1=/home/aagarwal/.tt-metal-cache-mxb-v1
C4=/home/aagarwal/.tt-metal-cache-mxb-v4tm
rm -rf "${C1:?}" "${C4:?}"; mkdir -p $C1 $C4
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
for p in a b; do
    echo "##### pass $p (job at ${SECONDS} s)"
    KA_ENV= run_cap mx${p}_v1_13129088_mux $Y $C1 400 $MAIN $L; [ $? = 2 ] && stop
    python3 $CON/swap.py apply fabric_exp/variants/v4tm $BK || stop
    run_cap mx${p}_v4tm_13129088_mux $Y $C4 400 $L; [ $? = 2 ] && stop
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo ALL_DONE
