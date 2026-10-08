#!/bin/bash
# t295 broker job: arm A then arm B, each in its own pytest process (one A/B pair).
T=/var/tmp/fasth3/t295
bash $T/run295b.sh A; ra=$?
bash $T/run295b.sh B; rb=$?
echo "[t295] rcA=$ra rcB=$rb"
for i in 0 1 2 3 4 5; do f=ltx_av_fast_1920x1088_$i.mp4
  a=$(md5sum < $T/outA/$f 2>/dev/null | cut -c1-32); b=$(md5sum < $T/outB/$f 2>/dev/null | cut -c1-32)
  echo "[t295] gen$i A=$a B=$b $([ -n "$a" ] && [ "$a" = "$b" ] && echo MATCH || echo DIFF)"
done
[ $ra = 0 ] && [ $rb = 0 ]
