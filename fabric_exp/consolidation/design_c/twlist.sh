#!/bin/bash
# twlist.sh TAG...: receiver-loop listings (design_a/rxlist.py --chain) of the profiled E1 images of each twin set
# tw_<TAG>_{tso,tv,hr,js1e} into /home/aagarwal/.tt-metal-v3r-builds/design_c/listings/<cfg>_<TAG>.lst, and a summary
# line per listing: instructions, loop instructions, loop sp accesses (non-save), mul count.
set -u
cd /home/aagarwal/tt-metal
D=/home/aagarwal/.tt-metal-v3r-builds/design_c
mkdir -p $D/listings
declare -A H=([tso]=16246661628456555625 [tv]=1960938547760765240 [hr]=16383455491875178035 [js1e]=1841379155958340204)
for tag in "$@"; do
  for c in tso tv hr js1e; do
    e=$D/tw_${tag}_$c/${H[$c]}/subordinate_active_erisc/subordinate_active_erisc.elf
    [ -f $e ] || e=$D/tw_${tag}_$c/${H[$c]}/active_erisc/active_erisc.elf
    [ -f $e ] || { echo "$c $tag: no elf"; continue; }
    o=$D/listings/${c}_${tag}.lst
    python3 fabric_exp/consolidation/design_a/rxlist.py $e --chain > $o
    printf "%-5s %-8s insns %4d loop %4d loopS %3d mul %d\n" $c $tag $(wc -l < $o) $(grep -c '^[0-9a-f]* L' $o) $(grep -c '^[0-9a-f]* LS' $o) $(grep -c ' mul ' $o)
  done
done
