#!/bin/bash
# E1 pass reshape, stage 1 (v4x1): the next completion chosen first, then one register group (packets-received, TX
# status, the completion's write-sent status), then acks, completion and frame, then the packet; no bounded waits.
# Stage 2 (v4x2): v4x1 + the packet's header word, routing fields and (one or two downstreams) local credit registers
# read as one group. Both data-checked, then every suite and the latency rows; v4u as the same-job control.
#   v4x_runs.sh [VARIANT[:LABEL] ...]   default: v4x1 v4x2 v4u:v4uxc (variants other than v4u are data-checked first)
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
for vl in ${@:-v4x1 v4x2 v4u:v4uxc}; do
    v=${vl%%:*}; l=${vl##*:}
    C=/home/aagarwal/.tt-metal-cache-$l
    rm -rf "${C:?}"; mkdir -p $C
    python3 $CON/swap.py apply fabric_exp/variants/$v $BK || { echo STOPPED; exit 1; }
    if [ $v != v4u ]; then
        run_cap ${l}_13129088_mesh $(suite_yaml mesh) $C 300 || { echo STOPPED; exit 1; }
        run_cap ${l}_13129088_val1d $(suite_yaml val1d_m1) $C 300 || { echo STOPPED; exit 1; }
        run_cap ${l}_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $C 120 || { echo STOPPED; exit 1; }
        run_cap ${l}_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $C 120 || { echo STOPPED; exit 1; }
    fi
    for s in 4352 mix flushmix flushmix4 burst 1d_m1 lat; do
        run_cap ${l}_13129088_$s $(suite_yaml $s) $C 200; [ $? = 2 ] && { echo STOPPED; exit 1; }
    done
    run_cap ${l}_13129088_ring fabric_exp/v4/exp_ring.yaml $C 120; [ $? = 2 ] && { echo STOPPED; exit 1; }
    run_cap ${l}_13129088_torusy fabric_exp/v4/exp_torus_y.yaml $C 120; [ $? = 2 ] && { echo STOPPED; exit 1; }
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
