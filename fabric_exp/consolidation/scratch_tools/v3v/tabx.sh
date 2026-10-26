#!/bin/bash
# tabx.sh VARIANT_DIR NAME: like stab5.sh but N/S with DECODE=x (destination path); cells instr/taken/loads/stall5
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v
V=$1; n=$2
cd $S/dis
for c in W NS 1D VC1; do for l in O3 O2 Os; do
  ( out="$c $l"; for sc in write fwd atomic fused fusedfl scatter compu compm; do
      t=$S/tmpx_${n}_${c}_${l}_$sc.txt
      if [ $c = NS ]; then DECODE=x ALIGN=al python3 ../rxscen.py $V ${n}_${l}_$c.dis ${n}_${l}_$c.elf $sc -v > $t; else ALIGN=al python3 ../rxscen.py $V ${n}_${l}_$c.dis ${n}_${l}_$c.elf $sc -v > $t; fi
      last=$(grep "^instr" $t | head -1)
      i=$(echo $last | awk '{print $2}'); tk=$(echo $last | awk '{print $12}' | sed 's/+1//'); ld=$(echo $last | awk '{print $4}')
      st=$(python3 ../stall.py $t 5 | awk '{print $4}')
      out="$out $sc $i/$tk/$ld/$st"
    done; echo "$out" ) &
done; done | sort -k1,1 -k2,2
wait
