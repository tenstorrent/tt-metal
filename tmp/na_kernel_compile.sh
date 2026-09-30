#!/bin/bash
# Off-device check: JIT-compile the neighborhood_sdpa compute kernel (trisc0/1/2) against a
# given worktree, using the exact commands from broker job 671. Usage: na_kernel_compile.sh <worktree>
WT=$(realpath "${1:-.}"); OUT=/tmp/t26kc; rc=0; i=0
GEN=/var/tmp/t10-tt-metal-cache/tt-metal-cache9268051462127607463/kernels/neighborhood_sdpa/17544978056465359982
rm -rf $OUT; mkdir -p $OUT/gen; cp $GEN/*.h $GEN/*.cpp $OUT/gen/; mkdir -p $OUT/gen/trisc{0,1,2}
sed -i "s#/home/smarton/fasth3/tt-metal/tt-project/worktrees/t10/#$WT/#" $OUT/gen/chlkc_*.cpp
while read -r cmd; do
  i=$((i+1))
  c=${cmd//\/home\/smarton\/fasth3\/tt-metal\/tt-project\/worktrees\/t10\/ttnn/$WT/ttnn}
  c=${c//\/home\/smarton\/fasth3\/tt-metal\/tt-project\/worktrees\/t10\/tt_metal\/hw\/firmware/$WT/tt_metal/hw/firmware}
  c=$(sed -E "s# -o [^ ]+# -o $OUT/k$i.o#; s# -MF [^ ]+# -MF $OUT/k$i.d#" <<<"$c")
  if (cd $OUT/gen/trisc0 && eval "$c") 2>$OUT/k$i.err; then echo "trisc cmd $i: OK"; else echo "trisc cmd $i: FAIL"; grep -m1 error: $OUT/k$i.err; rc=1; fi
done < "$(dirname "$0")/na_kernel_cmds.txt"
exit $rc
