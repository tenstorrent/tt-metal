#!/bin/bash
# Offline builds of the jitter edits at -O3 and -Os, PT_LOAD comparison against v3tk (ka13 builds).
cd /home/aagarwal/tt-metal
J=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/jitter
K=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/ka13
L=/home/aagarwal/tt-metal/fabric_exp/consolidation/scratch_tools/v3l/loadcmp.py
for o in O3 Os; do
  for d in fabric_exp/variants/v3tk_jit/*; do e=$(basename $d)
    r=$(python3 fabric_exp/consolidation/offline_build.py build --variant $d --out $J/${e}_$o --opt $o --jobs 32 | tail -1)
    c=$(python3 $L $J/${e}_$o $K/v3tk_$o active_erisc subordinate_active_erisc | tr '\n' ' ')
    echo "$o $e | ${r%% ELFs*} | $c"
  done
done
