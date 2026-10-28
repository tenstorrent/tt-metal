#!/bin/bash
# v4u attribution: v4ue0 = v4l + v4u's sender side (worker credit on a preset command buffer, grouped visit loads),
# v4ue1 = v4l + v4u's receiver side (grouped pass registers under first-level acks, single-slot completions). Each:
# ring and Y-torus validation, then ring, Y-torus, 4352 and mix rows. Then v4u again (same-job reference) and v4u's
# ring 1-hop PC profile (beside pcsv_v4f1r_ring1).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
V=fabric_exp/variants
for v in v4ue0 v4ue1; do
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
C=/home/aagarwal/.tt-metal-cache-v4u
python3 $CON/swap.py apply $V/v4u $BK || { echo STOPPED; exit 1; }
run_cap v4ur2_13129088_ring fabric_exp/v4/exp_ring.yaml $C 120
run_cap v4ur2_13129088_torusy fabric_exp/v4/exp_torus_y.yaml $C 120
run_cap v4ur2_13129088_4352 $(suite_yaml 4352) $C 120
run_cap v4ur2_13129088_mix $(suite_yaml mix) $C 300
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
bash $CON/pcsv_runs.sh $V/v4u v4ur Ring ring1
rm -f $M
