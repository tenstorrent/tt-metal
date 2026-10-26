#!/bin/bash
# Bursty senders (RESULTS §98): exp_burst.yaml (stream idle segments: bursts of N packets, then G idle cycles) at -O3
# for v3tk, v3v6, v3w4, v3x. At a fixed (N, G), any extra time across builds is router-caused. Labels p23_<variant>.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
for v in v3tk v3v6 v3w4 v3x; do build ${v}_burst $V/$v $C3 O3 p23_$v burst; done
rm -f $M
echo ALL_DONE
