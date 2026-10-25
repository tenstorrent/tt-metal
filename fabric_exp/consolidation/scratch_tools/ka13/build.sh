#!/bin/bash
# v3tk (keepalive as a host setting) vs v3t: offline builds at -O3/-O2/-Os, then PT_LOAD comparison of both ERISCs.
cd /home/aagarwal/tt-metal
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/ka13
V=fabric_exp/variants
for o in O3 O2 Os; do
  for v in v3t v3tk; do
    python3 fabric_exp/consolidation/offline_build.py build --variant $V/$v --out $S/${v}_$o --opt $o --jobs 24 | tail -1
  done
  echo "== $o: v3tk vs v3t"; python3 $S/../v3l/loadcmp.py $S/v3tk_$o $S/v3t_$o active_erisc subordinate_active_erisc
done
