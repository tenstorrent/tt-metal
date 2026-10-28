#!/bin/bash
# Main's router (stock: kernel v1, flags 0, firmware keepalive) at 1D sender depths that keep one channel a power of 2,
# to see where the 26 / 25 / 32 split's losses on main come from (job 076): worker 35 / forwarding 16, worker 19 /
# forwarding 32, worker 32 / forwarding 19 (receiver 32). v4u at worker 32 / forwarding 19 for comparison.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
L="TT_FABRIC_EXP_LOG_SLOTS=1"
MAIN="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
C=/home/aagarwal/.tt-metal-cache-main1d
for slots in 0,16,32 0,32,32 32,19,32; do
    for s in 1d ring; do
        y=$(suite_yaml 1d); [ $s = ring ] && y=fabric_exp/v4/exp_ring.yaml
        KA_ENV= run_cap main1d_s${slots//,/-}_$s $y $C 150 $MAIN TT_FABRIC_EXP_SLOTS_1D=$slots $L
        [ $? = 2 ] && { echo STOPPED; exit 1; }
    done
done
C=/home/aagarwal/.tt-metal-cache-v4u1d
python3 $CON/swap.py apply fabric_exp/variants/v4u $BK || { echo STOPPED; exit 1; }
run_cap v4u1d_s32-19-32_1d $(suite_yaml 1d_m1) $C 150 TT_FABRIC_EXP_SLOTS_1D=32,19,32 $L
run_cap v4u1d_s32-19-32_ring fabric_exp/v4/exp_ring.yaml $C 150 TT_FABRIC_EXP_SLOTS_1D=32,19,32 $L
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
