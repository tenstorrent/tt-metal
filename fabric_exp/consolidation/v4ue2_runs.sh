#!/bin/bash
# v4u attribution, second level: the sender half split (v4ue0a = preset command buffer only, v4ue0b = grouped visit
# loads only) and the receiver half split (v4ue1a = grouped pass registers only, v4ue1b = single-slot completions
# only). Each: ring and Y-torus validation, then ring, Y-torus, 4352 and mix rows.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
V=fabric_exp/variants
for v in v4ue0a v4ue0b v4ue1a v4ue1b; do
    C=/home/aagarwal/.tt-metal-cache-$v
    rm -rf "${C:?}"; mkdir -p $C
    python3 $CON/swap.py apply $V/$v $BK || { echo STOPPED; exit 1; }
    run_cap ${v}_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $C 120 || { echo STOPPED; exit 1; }
    run_cap ${v}_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $C 120 || { echo STOPPED; exit 1; }
    run_cap ${v}_13129088_ring fabric_exp/v4/exp_ring.yaml $C 120
    run_cap ${v}_13129088_torusy fabric_exp/v4/exp_torus_y.yaml $C 120
    run_cap ${v}_13129088_4352 $(suite_yaml 4352) $C 120
    run_cap ${v}_13129088_mix $(suite_yaml mix) $C 300
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
