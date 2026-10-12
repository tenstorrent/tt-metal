#!/bin/bash
# t452 (blx01, no device): builds /var/tmp/fasth3/t452/w, a symlink overlay of the t409/b tree (53f7b6b707 plus
# #449's uncommitted vsa kernel/host edits, built) whose two minimax_h3 Python dirs are real copies with sla.patch
# (04cbd225529 + ff705742990: SLA block selection) applied. Only Python differs, so t409/b's build is reused.
set -eo pipefail
F=/var/tmp/fasth3; B=$F/t409/b; W=$F/t452/w; P=$F/t452/sla.patch
[ ! -e $W ] || { echo "[mk452] $W exists"; exit 1; }
mkdir -p $W
# mirror <rel dir>: make $W/<rel> a real dir whose entries are symlinks into $B/<rel>
mirror() { mkdir -p $W/$1; for e in $B/$1/* $B/$1/.[!.]*; do [ -e "$e" ] || continue
  n=$(basename "$e"); [ -e $W/$1/$n ] || ln -s "$e" $W/$1/$n; done; }
mirror .
for leaf in models/tt_dit/models/transformers/minimax_h3 models/tt_dit/pipelines/minimax_h3; do
  p=""; for part in ${leaf//\// }; do
    q=${p:+$p/}$part; [ -L $W/$q ] && rm $W/$q
    if [ $q = $leaf ]; then cp -a $B/$q $W/$q; else mirror $q; fi; p=$q; done
done
cd $W && patch -p1 --no-backup-if-mismatch < $P
grep -q MINIMAX_H3_VSA_MODE models/tt_dit/pipelines/minimax_h3/pipeline_minimax_h3.py
echo "overlay of $B ($(git -C $B rev-parse --short HEAD) + uncommitted) + sla.patch $(md5sum < $P | cut -c1-12)" > $W/OVERLAY
cat $W/OVERLAY
