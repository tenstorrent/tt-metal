#!/bin/bash
# e0cmp.sh BASE_PREFIX EDIT_PREFIX [FN_REGEX]: compare sender_erisc functions (default run_main_loop) of the five hot E0
# configs between two e0hot.sh builds: same (identical code, same address) / moved / DIFF.
E=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/e0audit
B=$1; X=$2; FN=${3:-sender_erisc13run_main_loop}
for l in O3 O2 Os; do
  line="$(basename $X) $l:"
  for c in W:10733501891889981293 NS:8414232161770541884 1D:10744374302904868627 RING:16671252331734229980 VC1:1526198486225049198; do
    k=${c%%:*}; h=${c##*:}
    a=${B}_$l/$h/active_erisc/active_erisc.elf; b=${X}_$l/$h/active_erisc/active_erisc.elf
    r=$(python3 $E/tools/fncmp.py $a $b --fn "$FN" 2>&1 | grep -v '^main-loop' | awk '{print $1}' | sort | uniq -c | tr -s ' ' | tr '\n' ',')
    line="$line $k=[$r]"
  done
  echo "$line"
done
