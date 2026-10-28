#!/bin/bash
# v4 with no closures in the sender loop (forwarding dispatch, free-slots group read, round count and init recurse over
# the channel mask): validation (mesh, val1d without rings/mux/telemetry), the family-1 1D suite at worker depth 16 and
# 24, the 2D 4352 suite; then v4cuo (the same with the upstream credit out of line too) on the perf suites only. Fresh
# caches.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
C=/home/aagarwal/.tt-metal-cache-v4i
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4 $BK || { echo STOPPED; exit 1; }
run v4i_13129088_mesh $(suite_yaml mesh) $C O3 || { echo STOPPED; exit 1; }
run v4i_13129088_val1d $(suite_yaml val1d_m1) $C O3 || { echo STOPPED; exit 1; }
run v4iw16_13129088_1d $(suite_yaml 1d_m1) $C O3
EXTRA_ENV="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24" run v4iw24_13129088_1d $(suite_yaml 1d_m1) $C O3
run v4i_13129088_4352 $(suite_yaml 4352) $C O3
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
C=/home/aagarwal/.tt-metal-cache-v4cuo
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4cuo $BK || { echo STOPPED; exit 1; }
EXTRA_ENV="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24" run v4cuow24_13129088_1d $(suite_yaml 1d_m1) $C O3
run v4cuo_13129088_4352 $(suite_yaml 4352) $C O3
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
bash $CON/pcsv_runs.sh $V/v4 v4i2d 2D ew4352 hop3
