#!/bin/bash
# Offline builds for the round-11 merges and their parents at -O3/-O2/-Os, then exact-composition checks.
cd /home/aagarwal/tt-metal
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/merge11
V=fabric_exp/variants
for o in O3 O2 Os; do
  for v in v3l3e1c v3l3be1c v3l3 v3l3b v3e1c; do
    python3 fabric_exp/consolidation/offline_build.py build --variant $V/$v --out $S/${v}_$o --opt $o --jobs 24 | tail -1
  done
  for p in "v3l3e1c v3l3" "v3l3be1c v3l3b"; do set -- $p
    echo "== $o: $1 E0 vs $2"; python3 $S/../v3l/loadcmp.py $S/${1}_$o $S/${2}_$o active_erisc
    echo "== $o: $1 E1 vs v3e1c"; python3 $S/../v3l/loadcmp.py $S/${1}_$o $S/v3e1c_$o subordinate_active_erisc
  done
done
