#!/bin/bash
# Bursty senders for v3z (v3yb's E0 + v3x3's E1; RESULTS §98): exp_burst.yaml at -O3, with v3tk as the same-session
# reference. Labels p31_<variant>.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
for v in v3z v3tk; do build ${v}_burst31 $V/$v $C3 O3 p31_$v burst; done
rm -f $M
echo ALL_DONE
