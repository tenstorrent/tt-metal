#!/bin/bash
# 1D analysis (RESULTS §98): is 1D 1-hop bound by the worker credit loop? For v3q and v4: the 1D benchmark suite at the
# default worker depth (16), with Linear's spare slots given to the worker channel (35; TT_FABRIC_EXP_LINEAR_SPARE_SLOTS,
# a study host switch) and capped at 24; the 2D 4352 suite beside it; and a data-checked 1D validation at 35 (v4).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
for var in v3q v4; do
    C=/home/aagarwal/.tt-metal-cache-1d-$var
    rm -rf "${C:?}"; mkdir -p $C
    python3 $CON/swap.py apply $V/$var $BK || { echo STOPPED; exit 1; }
    run ${var}w16_13129088_1d $(suite_yaml 1d_m1) $C O3
    EXTRA_ENV="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1" run ${var}w35_13129088_1d $(suite_yaml 1d_m1) $C O3
    EXTRA_ENV="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24" run ${var}w24_13129088_1d $(suite_yaml 1d_m1) $C O3
    run ${var}2d_13129088_4352 $(suite_yaml 4352) $C O3
    if [ $var = v4 ]; then
        EXTRA_ENV="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1" run v4w35_13129088_val1d $(suite_yaml val1d_m1) $C O3
    fi
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
for f in fabric_exp/results/{v3q,v4}{w16,w35,w24}_13129088_1d.csv fabric_exp/results/{v3q,v4}2d_13129088_4352.csv; do
    [ -f $f ] && awk -F, -v f=$(basename $f .csv) 'NR>1 {printf "%-24s %-34s %.1f\n", f, $1, 1.35e9/$10}' $f
done
rm -f $M
echo ALL_DONE
