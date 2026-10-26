#!/bin/bash
# edits_build.sh: hot-config builds (all three levels) of every per-edit variant of v3sb and v3t.
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v
for b in v3sb v3t; do for c in templates os heartbeat param bittest init; do
  $S/hot.sh $S/v_${b}_$c $S/e_${b}_$c > /dev/null 2>&1
  echo "$b $c: $(ls $S/e_${b}_${c}_O3 $S/e_${b}_${c}_O2 $S/e_${b}_${c}_Os 2>/dev/null | grep -c '^[0-9]')"
done; done
