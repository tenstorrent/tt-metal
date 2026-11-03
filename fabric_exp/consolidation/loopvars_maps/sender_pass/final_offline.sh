#!/bin/bash
# final_offline.sh: the sender pass's last offline checks (V4_SENDER_OWNERSHIP.md §12): full coverage of v4txo6 (X6,
# against v4txo5b) and v4txo8 (X7+X8, against v4txo6) at -O3/-Os, the register-map guard per stage and of the final
# build against the lead (v4rxg3 under the job-464 host rule, hrnew), then the sender headroom check on v4txo8.
D=/home/aagarwal/tt-metal/fabric_exp/consolidation/loopvars_maps/sender_pass
cd /home/aagarwal/tt-metal
bash $D/bc_tx.sh v4txo6 "O3 Os" v4txo5b > $D/bc_v4txo6.log 2>&1
bash $D/bc_tx.sh v4txo8 "O3 Os" v4txo6 > $D/bc_v4txo8.log 2>&1
G=fabric_exp/consolidation/loopvars_maps/structure/guard_builds.py
for o in O3 Os; do
    timeout 3000 python3 $G v4txo5b_$o v4txo6_$o > $D/guard_v4txo5b_v4txo6_$o.txt 2>&1 &
    timeout 3000 python3 $G v4txo6_$o v4txo8_$o > $D/guard_v4txo6_v4txo8_$o.txt 2>&1 &
done
wait
for o in O3 Os; do
    while [ ! -s /home/aagarwal/.tt-metal-v3r-builds/cov_m/builds/hrnew_$o/results.json ]; do sleep 30; done
    timeout 3000 python3 $G hrnew_$o v4txo8_$o > $D/guard_hrnew_v4txo8_$o.txt 2>&1 &
done
[ -d fabric_exp/variants/v4txo8_tx1 ] || python3 fabric_exp/consolidation/loopvars_maps/sender_budget/synth_overlays_tx.py v4txo8
bash $D/hr_tx.sh v4txo8 > $D/hr_v4txo8.log 2>&1
wait
echo FINAL_OFFLINE_DONE
