#!/bin/bash
# v4s2 = v4s with only the send shared: forwarding channels with a packet send through one send (run-time loop,
# SenderChannel in their records); completions and credits stay per-channel compile-time code. Validation,
# 1D at 24 worker slots, 2D 4352, mix; v4j (frozen copy) re-run on 4352 and mix as the control; PC samples.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
W24="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24"
C=/home/aagarwal/.tt-metal-cache-v4s2
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4s2 $BK || { echo STOPPED; exit 1; }
run v4s2_13129088_mesh $(suite_yaml mesh) $C O3 || { echo STOPPED; exit 1; }
run v4s2_13129088_val1d $(suite_yaml val1d_m1) $C O3 || { echo STOPPED; exit 1; }
EXTRA_ENV="$W24" run v4s2w24_13129088_1d $(suite_yaml 1d_m1) $C O3
run v4s2_13129088_4352 $(suite_yaml 4352) $C O3
run v4s2_13129088_mix $(suite_yaml mix) $C O3
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
C=/home/aagarwal/.tt-metal-cache-v4jd
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4j $BK || { echo STOPPED; exit 1; }
run v4jd_13129088_4352 $(suite_yaml 4352) $C O3
run v4jd_13129088_mix $(suite_yaml mix) $C O3
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
bash $CON/pcsv_runs.sh $V/v4s2 v4s22d 2D fwd16 hop3 mcast
