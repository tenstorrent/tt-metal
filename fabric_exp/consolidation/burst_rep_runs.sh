#!/bin/bash
# Burst noise check (RESULTS §98): exp_burst.yaml again for v3tk, v3v6, v3x at -O3 (labels p25_<variant>), to see
# whether the cells that differ in p23 (bursts of 8 / 32 with 500-cycle gaps) repeat.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
for v in v3tk v3v6 v3x; do build ${v}_burst2 $V/$v $C3 O3 p25_$v burst; done
rm -f $M
echo ALL_DONE
