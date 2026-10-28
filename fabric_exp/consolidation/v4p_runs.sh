#!/bin/bash
# v4p = v4u + completions held for the next ack frame under first-level acks (at most 16 passes): at most one credit
# frame per packet when packets arrive slower than the receiver loop. Ring and torus only (mesh and 1D binaries are
# identical to v4u); then v4u as the control.
#   v4p_runs.sh [VARIANT[:LABEL] ...]   default: v4p v4u:v4uc (variants other than v4u get the ring and torus validation)
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
V=fabric_exp/variants
for vl in ${@:-v4p v4u:v4uc}; do
    v=${vl%%:*}; l=${vl##*:}
    C=/home/aagarwal/.tt-metal-cache-${v}-pb
    rm -rf "${C:?}"; mkdir -p $C
    python3 $CON/swap.py apply $V/$v $BK || { echo STOPPED; exit 1; }
    if [ $v != v4u ]; then
        run_cap ${v}_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $C 120 || { echo STOPPED; exit 1; }
        run_cap ${v}_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $C 120 || { echo STOPPED; exit 1; }
    fi
    run_cap ${l}_13129088_ring fabric_exp/v4/exp_ring.yaml $C 120
    run_cap ${l}_13129088_torusy fabric_exp/v4/exp_torus_y.yaml $C 120
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
