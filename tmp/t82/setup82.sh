#!/bin/bash
# blx03, #82: build the hoist-arm runtime root ~/fasth3/t82rt without a rebuild (the cherry-pick is
# JIT-header-only). Top-level entries symlink to the t48 tree; tt_metal/ is a symlink farm with the 15
# patched headers from hdrs82.tgz as real files. Both arms run t48's python and _ttnn.so; only
# TT_METAL_RUNTIME_ROOT (and the JIT cache) differs.
set -e
B=/home/smarton/fasth3/t48; R=/home/smarton/fasth3/t82rt; V=/var/tmp/fasth3/t82
test "$(git -C $B rev-parse --short=11 HEAD)" = a613d669eef || { echo "t48 not at a613d669eef"; exit 2; }
rm -rf $R; mkdir -p $R $V
for e in $B/* $B/.[!.]*; do n=$(basename $e); [ $n = tt_metal ] || ln -s $e $R/$n; done
cp -as $B/tt_metal $R/tt_metal
tar -tzf $V/hdrs82.tgz | grep -v /$ | while read f; do rm -f $R/$f; done
tar -xzf $V/hdrs82.tgz -C $R
mkdir -p $B/tmp/t82 && cp $V/test_block_sfpu_ab.py $V/run82.sh $V/drive82.sh $B/tmp/t82/
# Only the 15 headers may differ from the base tree.
n=$(diff -rq $B/tt_metal $R/tt_metal | tee $V/overlay_diff.txt | wc -l)
echo "SETUP82 overlay diffs=$n (want 15)"; cat $V/overlay_diff.txt
[ $n = 15 ]
