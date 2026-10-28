#!/bin/bash
# v4m = v4k with the whole 1D sparse-multicast case (write, advance, forward) out of line and v4j's two-way branch kept
# (v4k's run-time "write first" flag cost the 1D scatter rows 2-4, job 049). Validation, 1D at 24 worker slots, 2D 4352,
# mix; v4j re-run on 1D 24 as the control.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
W24="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24"
C=/home/aagarwal/.tt-metal-cache-v4m
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4 $BK || { echo STOPPED; exit 1; }
run v4m_13129088_mesh $(suite_yaml mesh) $C O3 || { echo STOPPED; exit 1; }
run v4m_13129088_val1d $(suite_yaml val1d_m1) $C O3 || { echo STOPPED; exit 1; }
EXTRA_ENV="$W24" run v4mw24_13129088_1d $(suite_yaml 1d_m1) $C O3
run v4m_13129088_4352 $(suite_yaml 4352) $C O3
run v4m_13129088_mix $(suite_yaml mix) $C O3
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
C=/home/aagarwal/.tt-metal-cache-v4jf
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4j $BK || { echo STOPPED; exit 1; }
EXTRA_ENV="$W24" run v4jfw24_13129088_1d $(suite_yaml 1d_m1) $C O3
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
