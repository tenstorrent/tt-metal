#!/bin/bash
# Drift control (RESULTS §98): v3tk's benchmark suites at -O3 now, against its runs of 09:00-18:00 (p12t, p13k,
# jit3_base). Label p15k.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
build v3tk_ctl_O3 $V/v3tk $C3 O3 p15k $BENCH
rm -f $M
echo ALL_DONE
