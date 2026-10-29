#!/bin/bash
# Speedy-path validation again (2026-10-29): job 155's validation failed on main's router as well (DATA_MISMATCH: the
# yaml had no flow control, so senders overran the receivers' buffers); v4/val_nbr.yaml now has enable_flow_control
# (vetted on the mock). Section 1 of spd_runs.sh with labels spdv2_<router>_<mode>.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
V=fabric_exp/v4/val_nbr.yaml
C1=/home/aagarwal/.tt-metal-cache-spd-v1
CS=/home/aagarwal/.tt-metal-cache-spd-v4spd
rm -rf "${C1:?}-two" "${C1:?}-one" "${CS:?}-two" "${CS:?}-one"; mkdir -p $C1-two $C1-one $CS-two $CS-one
main_env="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }

echo "##### 1. validation (job at ${SECONDS} s)"
KA_ENV= run_cap spdv2_v1_two $V $C1-two 300 $main_env; [ $? = 2 ] && stop
KA_ENV= run_cap spdv2_v1_one $V $C1-one 300 $main_env $E; [ $? = 2 ] && stop
python3 $CON/swap.py apply fabric_exp/variants/v4spd $BK || stop
run_cap spdv2_v4spd_two $V $CS-two 300; [ $? = 2 ] && stop
run_cap spdv2_v4spd_one $V $CS-one 300 $E; [ $? = 2 ] && stop
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }

rm -f $M
echo ALL_DONE
