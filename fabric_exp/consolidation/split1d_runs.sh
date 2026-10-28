#!/bin/bash
# The builder's 1D slot split (Linear, Ring on Blackhole: receiver keeps its tier depth, 32; the sender channels share
# the rest evenly, remainder to the worker channel: 26 / 25 / 32 of 83), on main's router and on v4u: data-checked at
# the new default, then each benchmark at the new default (_split) and at the tier depths (_tier,
# TT_FABRIC_EXP_TIER_SLOTS=1, 16 / 16 / 32). Main runs as stock: kernel v1, flags 0, the firmware keepalive.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
L="TT_FABRIC_EXP_LOG_SLOTS=1"
T=TT_FABRIC_EXP_TIER_SLOTS=1
MAIN="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
bench() {  # label-prefix yaml-suite yaml env...
    local p=$1 s=$2 y=$3 rc; shift 3
    run_cap ${p}_split_$s $y $C 150 "$@" $L; rc=$?; [ $rc = 2 ] && { echo STOPPED; exit 1; }
    run_cap ${p}_tier_$s $y $C 150 "$@" $L $T; rc=$?; [ $rc = 2 ] && { echo STOPPED; exit 1; }
}

C=/home/aagarwal/.tt-metal-cache-main1d
rm -rf "${C:?}"; mkdir -p $C
KA_ENV= run_cap main1d_split_val1d $(suite_yaml val1d_m1) $C 600 $MAIN $L || { echo STOPPED; exit 1; }
KA_ENV= run_cap main1d_split_val_ring fabric_exp/v4/val_ring_p150.yaml $C 400 $MAIN $L || { echo STOPPED; exit 1; }
KA_ENV= bench main1d 1d $(suite_yaml 1d) $MAIN
KA_ENV= bench main1d ring fabric_exp/v4/exp_ring.yaml $MAIN

C=/home/aagarwal/.tt-metal-cache-v4u1d
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply fabric_exp/variants/v4u $BK || { echo STOPPED; exit 1; }
run_cap v4u1d_split_val1d $(suite_yaml val1d_m1) $C 300 $L || { echo STOPPED; exit 1; }
run_cap v4u1d_split_val_ring fabric_exp/v4/val_ring_p150.yaml $C 300 $L || { echo STOPPED; exit 1; }
bench v4u1d 1d $(suite_yaml 1d_m1)
bench v4u1d ring fabric_exp/v4/exp_ring.yaml
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
