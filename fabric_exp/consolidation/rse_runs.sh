#!/bin/bash
# Rings on one ERISC without the mux (2026-10-28; mux-mode analysis, RESULTS §98): in mux mode v4tm matches main on
# every unicast row, is faster on linear multicast and slower on ring multicast (+6..8) and full-ring multicast (+21,
# any size). Question: is that the mux, or single-ERISC rings (first-level acks: v4's counter frames share the one TX
# queue with data)? exp_ring_se.yaml (exp_ring.yaml's rows and the full-ring multicast, 1024 and 4352 B) with
# TT_METAL_DISABLE_FABRIC_TWO_ERISC=1, main's router and v4se in the order main, v4se, main, v4se, hang capture armed.
# Labels rse<pass>_{v1,v4se}_13129088_rse; compare with same_job_compare.py rsea_v1 rsea_v4se rseb_v1 rseb_v4se.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
L="TT_FABRIC_EXP_LOG_SLOTS=1"
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
MAIN="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
Y=fabric_exp/v4/exp_ring_se.yaml
C1=/home/aagarwal/.tt-metal-cache-rse-v1
C4=/home/aagarwal/.tt-metal-cache-rse-v4se
rm -rf "${C1:?}" "${C4:?}"; mkdir -p $C1 $C4
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
for p in a b; do
    echo "##### pass $p (job at ${SECONDS} s)"
    KA_ENV= run_cap rse${p}_v1_13129088_rse $Y $C1 400 $MAIN $L $E; [ $? = 2 ] && stop
    python3 $CON/swap.py apply fabric_exp/variants/v4se $BK || stop
    run_cap rse${p}_v4se_13129088_rse $Y $C4 400 $L $E; [ $? = 2 ] && stop
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo ALL_DONE
