#!/bin/bash
# Main's router (stock: kernel v1, flags 0, firmware keepalive) at 1D depths that are all powers of 2: worker 16,
# forwarding 32, receiver 32 (80 of 83 slots). Data-checked first (1D 143, ring 19), then the 1D and ring rows, with the
# tier depths (16 / 16 / 32) again as the same-job control.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
L="TT_FABRIC_EXP_LOG_SLOTS=1"
MAIN="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
P2="TT_FABRIC_EXP_SLOTS_1D=16,32,32"
C=/home/aagarwal/.tt-metal-cache-main1d
KA_ENV= run_cap main1d_s16-32-32_val_1d $(suite_yaml val1d_m1) $C 600 $MAIN $P2 $L || { echo STOPPED; exit 1; }
KA_ENV= run_cap main1d_s16-32-32_val_ring fabric_exp/v4/val_ring_p150.yaml $C 400 $MAIN $P2 $L || { echo STOPPED; exit 1; }
for s in 1d ring; do
    y=$(suite_yaml 1d); [ $s = ring ] && y=fabric_exp/v4/exp_ring.yaml
    KA_ENV= run_cap main1d_s16-32-32_$s $y $C 150 $MAIN $P2 $L; [ $? = 2 ] && { echo STOPPED; exit 1; }
    KA_ENV= run_cap main1d_tierc_$s $y $C 150 $MAIN $L; [ $? = 2 ] && { echo STOPPED; exit 1; }
done
rm -f $M
