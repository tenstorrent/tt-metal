#!/bin/bash
# Speedy path on v4 (2026-10-29; variants/v4spd = v4vc2 + the host's speedy-path routers (ENABLE_SPEEDY_VC0, e.g. every
# router of a NeighborExchange fabric: VC0 worker-only, receiver forwarding off): the generic loop specializes to them,
# and VC0's receiver channel only delivers (no decode or admission), as main's speedy receiver. Offline: the 12
# NeighborExchange configs build (8 two-ERISC, 4 one-ERISC); every other config's image equals v4vc2's (752).
#   1. Validation, data-checked (v4/val_nbr.yaml: write, scatter, fused, atomic over every NeighborExchange link; vetted
#      on the mock), hang capture armed (a hang stops the job): main's router as shipped (kernel v1, flags 0, firmware
#      keepalive) first, two ERISCs and one; then v4spd, two ERISCs and one. Labels spdv_<router>_<mode>.
#   2. Throughput, same job: exp_1d_4352.yaml in full (its NeighborExchange write / scatter rows are the speedy path;
#      the other rows are 1D lines, v4spd = v4vc2 there): main as shipped, then v4spd (the study's settings). Labels
#      spda_<router>_13129088_1d.
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
KA_ENV= run_cap spdv_v1_two $V $C1-two 300 $main_env; [ $? = 2 ] && stop
KA_ENV= run_cap spdv_v1_one $V $C1-one 300 $main_env $E; [ $? = 2 ] && stop
python3 $CON/swap.py apply fabric_exp/variants/v4spd $BK || stop
run_cap spdv_v4spd_two $V $CS-two 300; [ $? = 2 ] && stop
run_cap spdv_v4spd_one $V $CS-one 300 $E; [ $? = 2 ] && stop
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }

echo "##### 2. throughput (job at ${SECONDS} s)"
KA_ENV= run_cap spda_v1_13129088_1d fabric_exp/exp_1d_4352.yaml $C1-two 300 $main_env; [ $? = 2 ] && stop
python3 $CON/swap.py apply fabric_exp/variants/v4spd $BK || stop
run_cap spda_v4spd_13129088_1d fabric_exp/exp_1d_4352.yaml $CS-two 300; [ $? = 2 ] && stop
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
