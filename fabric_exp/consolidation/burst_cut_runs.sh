#!/bin/bash
# Batch-cut test for the bursty 16 B regression (RESULTS §98; V3E1_NOTES §12.21): exp_burst.yaml at -O3 for v3x
# (same-session reference), v3x_nocut (control: no record-word cut) and v3x2 (cut only on passes with a packet),
# then the 2D/1D benchmark suites for v3x_nocut (what the cut is worth on steady rows). Labels p30_<variant>, p30nc.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
for v in v3x v3x_nocut v3x2; do build ${v}_burst3 $V/$v $C3 O3 p30_$v burst; done
build v3x_nocut_bench $V/v3x_nocut $C3 O3 p30nc $BENCH
rm -f $M
echo ALL_DONE
