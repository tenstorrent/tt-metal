#!/bin/bash
# v4j = v4i + receiver: no closures in the receiver's per-packet helpers, Route as the routing value tested on use (no
# packed word), downstream links held by value. Validation (mesh, val1d_m1), 1D at 24 worker slots, 2D 4352, and the
# mixed-traffic suite for v4j, v4i and v3q (no a09u32 baseline yet); then PC samples of the receiver-gap rows.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
W24="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24"
C=/home/aagarwal/.tt-metal-cache-v4j
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4 $BK || { echo STOPPED; exit 1; }
run v4j_13129088_mesh $(suite_yaml mesh) $C O3 || { echo STOPPED; exit 1; }
run v4j_13129088_val1d $(suite_yaml val1d_m1) $C O3 || { echo STOPPED; exit 1; }
EXTRA_ENV="$W24" run v4jw24_13129088_1d $(suite_yaml 1d_m1) $C O3
run v4j_13129088_4352 $(suite_yaml 4352) $C O3
run v4j_13129088_mix $(suite_yaml mix) $C O3
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
for v in v4i v3q; do
    C=/home/aagarwal/.tt-metal-cache-mix-$v
    rm -rf "${C:?}"; mkdir -p $C
    python3 $CON/swap.py apply $V/$v $BK || { echo STOPPED; exit 1; }
    run ${v}_13129088_mix $(suite_yaml mix) $C O3
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
bash $CON/pcsv_runs.sh $V/v4 v4j2d 2D hop3 mcast ewscatter
EXTRA_ENV="$W24" bash $CON/pcsv_runs.sh $V/v4 v4j1d 1D lin3
