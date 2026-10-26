#!/bin/bash
# stab5.sh VARIANT_DIR NAME [LAT]: per-path instructions / taken / loads / est. load-use stall (lat LAT) on the four hot E1 configs.
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v
V=$1; n=$2; LAT=${3:-5}
cd $S/dis
for c in W NS 1D VC1; do for l in O3 O2 Os; do
  ( out="$c $l"; for sc in write fwd atomic fused fusedfl scatter compu compm; do
      python3 ../rxscen.py $V ${n}_${l}_$c.dis ${n}_${l}_$c.elf $sc -v > /tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v/tmp_${n}_${c}_${l}_$sc.txt
      last=$(grep "^instr" /tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v/tmp_${n}_${c}_${l}_$sc.txt | head -1)
      i=$(echo $last | awk '{print $2}'); t=$(echo $last | awk '{print $12}' | sed 's/+1//'); ld=$(echo $last | awk '{print $4}')
      st=$(python3 ../stall.py /tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v/tmp_${n}_${c}_${l}_$sc.txt $LAT | awk '{print $4}')
      out="$out $sc $i/$t/$ld/$st"
    done; echo "$out" ) &
done; done | sort -k1,1 -k2,2
wait
