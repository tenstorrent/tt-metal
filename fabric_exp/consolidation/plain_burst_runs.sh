#!/bin/bash
# Bursty senders for the plain-source round (RESULTS §98): exp_burst.yaml at -O3 for v3zy (reference), v3p (the same
# design written plainly) and v3p_b1 (v3p without completion batching). Labels p34_<variant>.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
for v in v3zy v3p v3p_b1; do build ${v}_burst34 $V/$v $C3 O3 p34_$v burst; done
rm -f $M
echo ALL_DONE
