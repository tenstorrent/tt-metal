#!/bin/bash
# stab.sh BASE_NAME EDIT_PREFIX: main-loop region of each hot E1 config, base build vs base+edit, per edit and level.
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3t
B=$1; P=$2
cd $S/dis
for c in templates os heartbeat param bittest init; do
  for l in O3 O2 Os; do
    line="$c $l:"
    for k in W NS 1D VC1; do
      r=$(python3 ../fncmp.py ${B}_${l}_$k.elf ${P}_${c}_${l}_$k.elf | tail -1)
      st=$(echo "$r" | grep -o "instructions: [A-Z]*\|instructions: same" | awk '{print $2}')
      mv=$(echo "$r" | grep -o "starts 0x[0-9a-f]* -> 0x[0-9a-f]*" | awk '{print ($2==$4)?"":"@moved"}')
      line="$line $k=$st$mv"
    done
    echo "$line"
  done
done
