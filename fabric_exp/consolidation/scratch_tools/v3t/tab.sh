#!/bin/bash
# tab.sh VARIANT_DIR NAME: per-path counts on the four hot E1 configs at three levels (instr/taken(stack loads)).
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3t
V=$1; n=$2
cd $S/dis
for c in W NS 1D VC1; do for l in O3 O2 Os; do
  ( echo "$c $l $(python3 ../rxscen.py $V ${n}_${l}_$c.dis ${n}_${l}_$c.elf write fwd atomic fused fusedfl scatter compu compm | awk '{printf "%s %s/%s(sp%s) ", $1, $3, $13, $7}')" ) &
done; done | sort -k1,1 -k2,2 | sed 's/+1)//g'
wait
