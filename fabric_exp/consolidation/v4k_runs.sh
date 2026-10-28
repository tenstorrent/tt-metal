#!/bin/bash
# v4k = v4j + readability round: channels_with_packets named and documented, the 1D sparse-multicast hop out of line,
# downstreams named by direction (no action_to_slots). Senders byte-identical to v4j. Validation,
# 1D at 24 worker slots, 2D 4352, mix; v4j (frozen copy) re-run on 4352 and mix as the control; PC samples.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
W24="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24"
C=/home/aagarwal/.tt-metal-cache-v4k
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4 $BK || { echo STOPPED; exit 1; }
run v4k_13129088_mesh $(suite_yaml mesh) $C O3 || { echo STOPPED; exit 1; }
run v4k_13129088_val1d $(suite_yaml val1d_m1) $C O3 || { echo STOPPED; exit 1; }
EXTRA_ENV="$W24" run v4kw24_13129088_1d $(suite_yaml 1d_m1) $C O3
run v4k_13129088_4352 $(suite_yaml 4352) $C O3
run v4k_13129088_mix $(suite_yaml mix) $C O3
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
C=/home/aagarwal/.tt-metal-cache-v4je
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4j $BK || { echo STOPPED; exit 1; }
run v4je_13129088_4352 $(suite_yaml 4352) $C O3
run v4je_13129088_mix $(suite_yaml mix) $C O3
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
bash $CON/pcsv_runs.sh $V/v4 v4k2d 2D hop3 mcast
EXTRA_ENV="$W24" bash $CON/pcsv_runs.sh $V/v4 v4k1d 1D lin3 linmcast
