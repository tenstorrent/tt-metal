#!/bin/bash
# Early completion-check diagnostics (RESULTS §98; V3E1_NOTES §12.16). DIAGNOSTIC builds v3tk_cw / v3vc_cw (never for
# shipping) against same-session v3tk / v3vc, benchmarks at -O3. Labels p19k, p19kcw, p19c, p19ccw.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
build v3tk_cwref_O3 $V/v3tk $C3 O3 p19k $BENCH
build v3tk_cw_O3 $V/v3tk_cw $C3 O3 p19kcw $BENCH
build v3vc_cwref_O3 $V/v3vc $C3 O3 p19c $BENCH
build v3vc_cw_O3 $V/v3vc_cw $C3 O3 p19ccw $BENCH
rm -f $M
echo ALL_DONE
