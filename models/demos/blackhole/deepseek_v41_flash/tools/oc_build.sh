#!/bin/bash
# Private relink build of ONLY the unified_routed_expert_ffn op: compiles the op's unity TU from the pf_onecopy worktree sources with the main
# build's compile flags (read-only use of the main build dir), relinks a private _ttnncpp.so into $B/lib. Use with LD_LIBRARY_PATH=$B/lib.
set -e
M=/mnt/tt-data/ssinghal/tests/tt-metal; W=/mnt/tt-data/ssinghal/wt/pf_onecopy; B=/mnt/tt-data/ssinghal/wt/pf_onecopy_build
OP=ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn
mkdir -p $B/lib $B/obj
sed "s#$M/ttnn/cpp/#$W/ttnn/cpp/#g" $M/build_Release/$OP/CMakeFiles/ttnn_op_experimental_deepseek_prefill_unified_routed_expert_ffn.dir/Unity/unity_0_cxx.cxx > $B/obj/unity_0_cxx.cxx
cd $M/build_Release
CC=$(cat /mnt/tt-data/ssinghal/wt/pf_onecopy_build/oc_compile_cmd.txt)
# worktree headers first, private object / source, no depfile
CC=$(echo "$CC" | sed -e "s#-I$M/ttnn/cpp #-I$W/ttnn/cpp -I$M/ttnn/cpp #" -e "s#-I$M/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn#-I$W/$OP#" -e 's# -MD##' -e 's# -MT [^ ]*##' -e 's# -MF [^ ]*##' -e "s# -o [^ ]*# -o $B/obj/unity_0_cxx.cxx.o#" -e "s# -c [^ ]*# -c $B/obj/unity_0_cxx.cxx#")
echo "[oc_build] compiling"; time eval "$CC"
L=$(cat /mnt/tt-data/ssinghal/wt/pf_onecopy_build/oc_link_cmd.txt)
OBJ=$OP/CMakeFiles/ttnn_op_experimental_deepseek_prefill_unified_routed_expert_ffn.dir/Unity/unity_0_cxx.cxx.o
L=$(echo "$L" | sed -e "s# $OBJ # $B/obj/unity_0_cxx.cxx.o #" -e 's# -Xlinker --dependency-file=[^ ]*##' -e "s# -o ttnn/_ttnncpp.so # -o $B/lib/_ttnncpp.so #" -e 's#^: && ##' -e 's# && :$##')
echo "$L" | grep -q "$B/obj/unity_0_cxx.cxx.o" || { echo "object substitution failed"; exit 2; }
echo "[oc_build] linking"; time eval "$L"
ls -la $B/lib
