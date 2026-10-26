#!/bin/bash
# Worker-window slack (RESULTS §98): v3v6, v3w4, v3w at -O3 with TT_FABRIC_EXP_WORKER_SLOTS=20 (worker channel 20
# slots, each forwarding channel one fewer). Benchmarks; labels p21s_<variant>. Study knob only: measures whether
# window slack removes the credit-latency sensitivity.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
export EXTRA_ENV="TT_FABRIC_EXP_WORKER_SLOTS=20"
for v in v3v6 v3w4 v3w; do build ${v}_slots20 $V/$v $C3 O3 p21s_$v $BENCH; done
rm -f $M
echo ALL_DONE
