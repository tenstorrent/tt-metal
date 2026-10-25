#!/bin/bash
# runtab.sh OUT VARIANTS... : scenario table over W/NS/1D x O3/O2/Os
out=$1; shift
cd /tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3e1c/dis
for c in W NS 1D; do for l in O3 O2 Os; do for v in "$@"; do
  python3 ../scen.py $v ${v}_${l}_$c.dis ${v}_${l}_$c.elf write scatter atomic fused fusedfl -d 2>&1 | awk -v t="$c $l" '/^   HIT/{h++} /^    (instr|no|ERR)/{d=$0} /^v3/{printf "%-6s %s %-40s | nocon:%s hits:%d\n", t, $1" "$2, substr($0,16,60), d, h; h=0}' &
done; done; done | sed 's/loads.*taken/taken/g' > $out
wait
