#!/bin/bash
# Bursty senders for the E0 candidates (RESULTS §98; E0_AUDIT.md): exp_burst.yaml at -O3, labels p28_<variant>.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
for v in v3yc v3v6l3 v3yb; do build ${v}_burst $V/$v $C3 O3 p28_$v burst; done
rm -f $M
echo ALL_DONE
