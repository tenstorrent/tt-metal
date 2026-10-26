#!/bin/bash
# edtab.sh VARIANT_SRC_PREFIX BUILD_PREFIX: path tables of every per-edit build (source dirs v_<x>_<edit>).
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v
V=$1; P=$2
for c in templates os heartbeat param bittest init; do
  $S/tab.sh $S/v_${V}_$c ${P}_$c > $S/edtab_${P}_$c.txt &
done
wait
