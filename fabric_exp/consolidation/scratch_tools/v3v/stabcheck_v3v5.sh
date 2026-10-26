#!/bin/bash
# stabcheck.sh BASE_PREFIX EDIT_DIR_ROOT EDIT...: build the four hot E1 configs of each edit at -O3/-O2/-Os and compare
# run_main_loop's main-loop region against the base builds (dis/<BASE_PREFIX>_<lvl>_<cfg>.elf).
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v
B=$1; R=$2; shift 2
cd /home/aagarwal/tt-metal
for e in "$@"; do
  for l in O3 O2; do python3 $S/ob.py build --variant $R/$e --out $S/st_v3v5_${e}_$l --opt $l --hashes 10856749301170935304,1539970155337268356,10388088869406661095,9530214107180203457 --jobs 8 >/dev/null 2>&1; done
  python3 $S/ob.py build --variant $R/$e --out $S/st_v3v5_${e}_Os --opt Os --hashes 10856749301170935304,1539970155337268356,10388088869406661095 --jobs 8 >/dev/null 2>&1
  python3 $S/ob.py build --variant $R/$e --out $S/st_v3v5_${e}_Os --opt O2 --set-opt Os --hashes 9530214107180203457 --jobs 8 >/dev/null 2>&1
  for l in O3 O2 Os; do
    line="$e $l:"
    for c in W:10856749301170935304 NS:1539970155337268356 1D:10388088869406661095 VC1:9530214107180203457; do
      k=${c%%:*}; h=${c##*:}
      r=$(python3 $S/fncmp.py $S/dis/${B}_${l}_$k.elf $S/st_v3v5_${e}_$l/$h/subordinate_active_erisc/subordinate_active_erisc.elf 2>&1 | grep "^main-loop region" | python3 -c "
import sys,re
l=sys.stdin.read()
m=re.search(r'instructions: (\w+), starts (\S+) -> (\S+)',l)
print('none' if not m else ('DIFF' if m.group(1)=='DIFF' else ('same' if m.group(2)==m.group(3) else 'moved')))")
      line="$line $k=$r"
    done
    echo "$line"
  done
done
