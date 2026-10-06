#!/bin/bash
# Rebuild ONLY the unified_routed_expert_ffn op of a tree's build and relink its _ttnncpp.so, safely for running jobs (they keep the old inode mapped):
# the new library is written under a temp name in the same directory and renamed over the old one; the old file is kept as _ttnncpp.so.pre_<tag>.
# usage: oc_main_rebuild.sh <tree> <tag>      (tree = /mnt/tt-data/ssinghal/tests/tt-metal for the main build; sources are taken from <tree>)
# It does NOT run ninja/cmake (a ninja run would re-run cmake and rebuild everything stale); it replays the op's unity-TU compile and the _ttnncpp link command
# that ninja recorded (`ninja -t commands`, saved in oc_compile_cmd.txt / oc_link_cmd.txt next to this script's build dir) with the object swapped.
set -e
M=${1:?tree}; TAG=${2:?tag}; B=/mnt/tt-data/ssinghal/wt/pf_onecopy_build
OP=ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn
OBJ=$OP/CMakeFiles/ttnn_op_experimental_deepseek_prefill_unified_routed_expert_ffn.dir/Unity/unity_0_cxx.cxx.o
cd $M/build_Release
T=.tmp_ring_$$
CC=$(cat $B/oc_compile_cmd.txt)
CC=$(echo "$CC" | sed -e 's# -MD##' -e 's# -MT [^ ]*##' -e 's# -MF [^ ]*##' -e "s# -o [^ ]*# -o $M/build_Release/$OBJ.$T#")
echo "[rebuild] compile"; eval "$CC"
L=$(cat $B/oc_link_cmd.txt)
L=$(echo "$L" | sed -e "s# $OBJ # $OBJ.$T #" -e 's# -Xlinker --dependency-file=[^ ]*##' -e "s# -o ttnn/_ttnncpp.so # -o ttnn/_ttnncpp.so.$T #" -e 's#^: && ##' -e 's# && :$##')
echo "$L" | grep -q "$OBJ.$T" || { echo "object substitution failed"; exit 2; }
echo "[rebuild] link"; eval "$L"
strings -a ttnn/_ttnncpp.so.$T | grep -q "ring weights mode needs 8 live DRAM banks" || { echo "new library lacks ring mode"; exit 3; }
cp ttnn/_ttnncpp.so.$T lib/_ttnncpp.so.$T
# keep the old files (hard links keep the old inodes), then atomic renames
for f in lib/_ttnncpp.so ttnn/_ttnncpp.so; do [ -e $f.pre_$TAG ] || ln $f $f.pre_$TAG; done
[ -e $OBJ.pre_$TAG ] || ln $OBJ $OBJ.pre_$TAG
mv -f lib/_ttnncpp.so.$T lib/_ttnncpp.so
mv -f ttnn/_ttnncpp.so.$T ttnn/_ttnncpp.so
mv -f $OBJ.$T $OBJ
ls -la lib/_ttnncpp.so* ttnn/_ttnncpp.so*
