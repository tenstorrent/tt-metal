#!/bin/bash
# v4f3 = v4f2 (one credit frame per pass, completions batched under acks) + transaction ids in contiguous slot blocks
# (an id comes back a ring later): validation everywhere, then the benchmark suites. Hang capture armed.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
V=fabric_exp/variants
W24="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24"
v=v4f3
C=/home/aagarwal/.tt-metal-cache-$v
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/$v $BK || { echo STOPPED; exit 1; }
run_cap ${v}_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $C 120 || { echo STOPPED; exit 1; }
run_cap ${v}_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $C 120 || { echo STOPPED; exit 1; }
run_cap ${v}_13129088_mesh $(suite_yaml mesh) $C 300 || { echo STOPPED; exit 1; }
run_cap ${v}_13129088_val1d $(suite_yaml val1d_m1) $C 300 || { echo STOPPED; exit 1; }
run_cap ${v}_13129088_ring fabric_exp/v4/exp_ring.yaml $C 120
run_cap ${v}_13129088_torusy fabric_exp/v4/exp_torus_y.yaml $C 120
run_cap ${v}_13129088_4352 $(suite_yaml 4352) $C 120
EXTRA_ENV="$W24" run_cap ${v}w24_13129088_1d $(suite_yaml 1d_m1) $C 120
run_cap ${v}_13129088_mix $(suite_yaml mix) $C 300
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
