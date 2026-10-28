#!/bin/bash
# v4s = v4j with one send and one forwarding visit: the channel's addresses as a SenderChannel value (worker:
# compile-time constants; forwarding channels: from their records, one visit reached by a run-time loop). Validation,
# 1D at 24 worker slots, 2D 4352, mix; v4j (frozen copy) re-run on 4352 and mix as the control; PC samples.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
W24="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24"
C=/home/aagarwal/.tt-metal-cache-v4s
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4 $BK || { echo STOPPED; exit 1; }
run v4s_13129088_mesh $(suite_yaml mesh) $C O3 || { echo STOPPED; exit 1; }
run v4s_13129088_val1d $(suite_yaml val1d_m1) $C O3 || { echo STOPPED; exit 1; }
EXTRA_ENV="$W24" run v4sw24_13129088_1d $(suite_yaml 1d_m1) $C O3
run v4s_13129088_4352 $(suite_yaml 4352) $C O3
run v4s_13129088_mix $(suite_yaml mix) $C O3
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
C=/home/aagarwal/.tt-metal-cache-v4jc
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4j $BK || { echo STOPPED; exit 1; }
run v4jc_13129088_4352 $(suite_yaml 4352) $C O3
run v4jc_13129088_mix $(suite_yaml mix) $C O3
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
bash $CON/pcsv_runs.sh $V/v4 v4s2d 2D fwd16 hop3 mcast
