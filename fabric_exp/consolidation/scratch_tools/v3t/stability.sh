#!/bin/bash
# stability.sh VARIANT_DIR NAME: derive the six edits from VARIANT_DIR, build base and edits on the hot configs, and
# compare the main-loop region and the per-path counts.
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3t
V=$1; N=$2
cd /home/aagarwal/tt-metal
rm -rf "${S:?}/j_${N:?}"
python3 fabric_exp/consolidation/v3r_renames.py join $V $S/j_$N
$S/hot.sh $V $S/$N > /dev/null 2>&1 &
for c in templates os heartbeat param bittest init; do
  rm -rf "${S:?}/k_${N:?}_$c" "${S:?}/v_${N:?}_$c"
  python3 $S/derive_edits.py $S/j_$N $S/k_${N}_$c $c > /dev/null
  python3 fabric_exp/consolidation/v3r_renames.py split $S/k_${N}_$c $S/v_${N}_$c
  rm -rf "${S:?}/k_${N:?}_$c"
done
wait
for c in templates os heartbeat param bittest init; do $S/hot.sh $S/v_${N}_$c $S/e_${N}_$c > /dev/null 2>&1; done
$S/stab.sh $N e_$N > $S/stab_$N.txt
