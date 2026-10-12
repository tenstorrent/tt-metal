#!/bin/bash
# t455 (blx01, no device): /var/tmp/fasth3/t455/w, a symlink overlay of t452/w whose minimax_h3 test dir is a real
# dir (symlinks into t452/w) holding test_sla_device_minimax_h3.py from $T/.
set -eo pipefail
F=/var/tmp/fasth3; B=$F/t452/w; T=$F/t455; W=$T/w; D=models/tt_dit/tests/models/minimax_h3
[ ! -e $W ] || { echo "[mk455] $W exists"; exit 1; }
mirror() { mkdir -p $W/$1; for e in $B/$1/* $B/$1/.[!.]*; do [ -e "$e" ] || continue
  n=$(basename "$e"); [ -e $W/$1/$n ] || ln -s "$e" $W/$1/$n; done; }
p=""; for part in . ${D//\// }; do q=${p:+$p/}$part; [ $q = . ] && q=.; [ -L $W/$q ] && rm $W/$q; mirror $q; [ $q = . ] && q=""; p=$q; done
cp $T/test_sla_device_minimax_h3.py $W/$D/
echo "$(cat $B/OVERLAY) + t455 test $(md5sum < $T/test_sla_device_minimax_h3.py | cut -c1-12)" > $W/OVERLAY
cat $W/OVERLAY
