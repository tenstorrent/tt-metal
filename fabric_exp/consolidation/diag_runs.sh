#!/bin/bash
# E1 idle-polling diagnostics (RESULTS §98; V3E1_NOTES §12.15). DIAGNOSTIC builds, never for shipping: after a pass
# with no progress, E1 spins N cycles on its own cycle counter. Benchmarks at -O3, with same-session references.
# Labels: p17c (v3vc), p17d8/p17d20/p17d50/p17d100 (v3vc_dN), p17k (v3tk), p17kd50 (v3tk_d50).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
build v3vc_ref_O3 $V/v3vc $C3 O3 p17c $BENCH
for n in 8 20 50 100; do build v3vc_d${n}_O3 $V/v3vc_d$n $C3 O3 p17d$n $BENCH; done
build v3tk_ref_O3 $V/v3tk $C3 O3 p17k $BENCH
build v3tk_d50_O3 $V/v3tk_d50 $C3 O3 p17kd50 $BENCH
rm -f $M
echo ALL_DONE
