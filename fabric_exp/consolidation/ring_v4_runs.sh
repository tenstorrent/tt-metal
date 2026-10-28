#!/bin/bash
# v4 with bubble flow control and first-level acks: ring / Y-torus validation (data-checked; the tests main passes on
# this P150x8, job 058), then their benchmark rows for v4 and v3q (v3q runs its generic loop on these configs).
# Hang capture armed (runs take ~10 s; 120 s is a hang).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
V=fabric_exp/variants
C=/home/aagarwal/.tt-metal-cache-v4r
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4 $BK || { echo STOPPED; exit 1; }
run_cap v4r_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $C 120 || { echo STOPPED; exit 1; }
run_cap v4r_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $C 120 || { echo STOPPED; exit 1; }
run_cap v4r_13129088_ring fabric_exp/v4/exp_ring.yaml $C 120
run_cap v4r_13129088_torusy fabric_exp/v4/exp_torus_y.yaml $C 120
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
C=/home/aagarwal/.tt-metal-cache-v3q-ring
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v3q $BK || { echo STOPPED; exit 1; }
run_cap v3q_13129088_ring fabric_exp/v4/exp_ring.yaml $C 120
run_cap v3q_13129088_torusy fabric_exp/v4/exp_torus_y.yaml $C 120
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
