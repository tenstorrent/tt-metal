#!/bin/bash
# Is the batched completion path worth its complexity now that E0's credits are cheap (v4u)? v4sc1 = v4u with
# single-slot completions on every VC (the path rings already use); v4sc2 = v4sc1 with that path checking a slot's
# writes once (a later pass credits it) instead of waiting up to 32 checks. Data-checked first, then every suite;
# v4u as the same-job control. 1D runs get the builder's 26 / 25 / 32 split (all three alike).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
for v in v4sc1 v4sc2 v4u; do
    l=$v; [ $v = v4u ] && l=v4uctl
    C=/home/aagarwal/.tt-metal-cache-$l
    rm -rf "${C:?}"; mkdir -p $C
    python3 $CON/swap.py apply fabric_exp/variants/$v $BK || { echo STOPPED; exit 1; }
    if [ $v != v4u ]; then
        run_cap ${l}_13129088_mesh $(suite_yaml mesh) $C 300 || { echo STOPPED; exit 1; }
        run_cap ${l}_13129088_val1d $(suite_yaml val1d_m1) $C 300 || { echo STOPPED; exit 1; }
        run_cap ${l}_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $C 120 || { echo STOPPED; exit 1; }
        run_cap ${l}_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $C 120 || { echo STOPPED; exit 1; }
    fi
    for s in 4352 mix flushmix flushmix4 burst 1d_m1; do
        run_cap ${l}_13129088_$s $(suite_yaml $s) $C 150; [ $? = 2 ] && { echo STOPPED; exit 1; }
    done
    run_cap ${l}_13129088_ring fabric_exp/v4/exp_ring.yaml $C 120; [ $? = 2 ] && { echo STOPPED; exit 1; }
    run_cap ${l}_13129088_torusy fabric_exp/v4/exp_torus_y.yaml $C 120; [ $? = 2 ] && { echo STOPPED; exit 1; }
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
